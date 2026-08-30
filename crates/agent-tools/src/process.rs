use crate::{CommandLimits, ToolContext, ToolError, ToolResult};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::{ExitStatus, Stdio};
use std::time::Instant;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};

pub(crate) async fn run_process(
    context: &ToolContext,
    program: &str,
    args: &[String],
    cwd: &Path,
    environment: &BTreeMap<String, String>,
    limits: &CommandLimits,
) -> Result<ToolResult, ToolError> {
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(cwd)
        .envs(environment)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);

    let mut child = command
        .spawn()
        .map_err(|error| ToolError::Execution(error.to_string()))?;
    let process_id = child.id();
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ToolError::Execution("stdout unavailable".to_owned()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| ToolError::Execution("stderr unavailable".to_owned()))?;
    let stdout_task = tokio::spawn(read_stream_limited(stdout, limits.stdout_limit_bytes));
    let stderr_task = tokio::spawn(read_stream_limited(stderr, limits.stderr_limit_bytes));
    let started = Instant::now();

    let (status, outcome) = tokio::select! {
        result = child.wait() => (Some(result.map_err(|error| ToolError::Execution(error.to_string()))?), "completed"),
        () = tokio::time::sleep(std::time::Duration::from_secs(limits.timeout_seconds)) => {
            terminate_process_tree(&mut child, process_id).await;
            (child.wait().await.ok(), "timeout")
        },
        () = context.cancellation.cancelled() => {
            terminate_process_tree(&mut child, process_id).await;
            (child.wait().await.ok(), "cancelled")
        },
    };

    let (stdout, stdout_truncated) = stdout_task
        .await
        .map_err(|error| ToolError::Execution(error.to_string()))??;
    let (stderr, stderr_truncated) = stderr_task
        .await
        .map_err(|error| ToolError::Execution(error.to_string()))??;
    let success = outcome == "completed" && status.as_ref().is_some_and(ExitStatus::success);
    let exit_code = status.and_then(|value| value.code());
    Ok(ToolResult {
        content: json!({
            "outcome": outcome,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }),
        summary: match outcome {
            "timeout" => format!("command timed out after {} seconds", limits.timeout_seconds),
            "cancelled" => "command cancelled".to_owned(),
            _ if success => "command succeeded".to_owned(),
            _ => format!(
                "command failed with exit code {}",
                exit_code.map_or_else(|| "unknown".to_owned(), |code| code.to_string())
            ),
        },
        truncated: stdout_truncated || stderr_truncated,
        metadata: json!({
            "kind":"command",
            "outcome":outcome,
            "success":success,
            "exit_code":exit_code,
            "duration_ms":started.elapsed().as_millis(),
            "stdout_truncated":stdout_truncated,
            "stderr_truncated":stderr_truncated,
        }),
    })
}

async fn terminate_process_tree(child: &mut Child, process_id: Option<u32>) {
    #[cfg(unix)]
    if let Some(process_id) = process_id {
        let process_group = format!("-{process_id}");
        let _ = Command::new("kill")
            .args(["-TERM", "--", &process_group])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let _ = Command::new("kill")
            .args(["-KILL", "--", &process_group])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
    }

    #[cfg(windows)]
    if let Some(process_id) = process_id {
        let process_id = process_id.to_string();
        let _ = Command::new("taskkill")
            .args(["/PID", &process_id, "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
    }

    let _ = child.kill().await;
}

async fn read_stream_limited<R: AsyncRead + Unpin>(
    mut reader: R,
    limit: usize,
) -> Result<(String, bool), ToolError> {
    let mut head = Vec::new();
    let mut tail = Vec::new();
    let head_limit = limit / 2;
    let tail_limit = limit.saturating_sub(head_limit);
    let mut buffer = [0_u8; 8192];
    let mut total = 0_usize;
    loop {
        let count = reader
            .read(&mut buffer)
            .await
            .map_err(|error| ToolError::Io(error.to_string()))?;
        if count == 0 {
            break;
        }
        total = total.saturating_add(count);
        let head_available = head_limit.saturating_sub(head.len());
        let to_head = count.min(head_available);
        head.extend_from_slice(&buffer[..to_head]);
        if to_head < count && tail_limit > 0 {
            tail.extend_from_slice(&buffer[to_head..count]);
            if tail.len() > tail_limit {
                tail.drain(..tail.len() - tail_limit);
            }
        }
    }
    let truncated = total > limit;
    if truncated && !head.is_empty() && !tail.is_empty() {
        head.extend_from_slice(b"\n... output truncated; tail follows ...\n");
    }
    head.extend_from_slice(&tail);
    Ok((String::from_utf8_lossy(&head).into_owned(), truncated))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::{CommandProfiles, ExecutionLimits};
    use agent_security::WorkspaceGuard;
    use std::fs;
    use std::process::Command as StdCommand;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn timeout_terminates_descendant_process_group() -> Result<(), Box<dyn std::error::Error>>
    {
        let root = tempfile::tempdir()?;
        let context = ToolContext {
            session_id: agent_security::SessionId::new(),
            task_id: agent_security::TaskId::new(),
            call_id: agent_security::ToolCallId::new(),
            workspace: WorkspaceGuard::new(root.path())?,
            cancellation: CancellationToken::new(),
            limits: ExecutionLimits::default(),
        };
        let limits = CommandLimits {
            timeout_seconds: 1,
            stdout_limit_bytes: CommandProfiles::default().default.stdout_limit_bytes,
            stderr_limit_bytes: CommandProfiles::default().default.stderr_limit_bytes,
        };
        let result = run_process(
            &context,
            "sh",
            &[
                "-c".to_owned(),
                "sleep 30 & echo $! > child.pid; wait".to_owned(),
            ],
            root.path(),
            &BTreeMap::new(),
            &limits,
        )
        .await?;
        assert_eq!(result.content["outcome"], "timeout");
        let child_id = fs::read_to_string(root.path().join("child.pid"))?;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let output = StdCommand::new("kill")
            .args(["-0", child_id.trim()])
            .output()?;
        assert!(!output.status.success());
        Ok(())
    }
}
