use crate::{Tool, ToolContext, ToolError, ToolRegistry, ToolResult};
use agent_model::ToolDefinition;
use agent_security::RiskLevel;
use async_trait::async_trait;
use globset::{Glob, GlobSetBuilder};
use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use walkdir::WalkDir;

pub fn register_builtin_tools(registry: &mut ToolRegistry) -> Result<(), ToolError> {
    registry.register(ListDirectory)?;
    registry.register(ReadFile)?;
    registry.register(ReadFileRange)?;
    registry.register(GlobFiles)?;
    registry.register(Grep)?;
    registry.register(GitStatus)?;
    registry.register(GitDiff)?;
    registry.register(PatchFile)?;
    registry.register(WriteFile)?;
    registry.register(RunCommand)?;
    Ok(())
}

fn schema(name: &str, description: &str, properties: Value, required: &[&str]) -> ToolDefinition {
    ToolDefinition::function(
        name,
        description,
        json!({
            "type":"object", "properties":properties, "required":required, "additionalProperties":false
        }),
    )
}

fn decode<T: for<'de> Deserialize<'de>>(value: Value) -> Result<T, ToolError> {
    serde_json::from_value(value).map_err(|error| ToolError::InvalidArguments(error.to_string()))
}

fn validate_decode<T: for<'de> Deserialize<'de>>(value: &Value) -> Result<(), ToolError> {
    serde_json::from_value::<T>(value.clone())
        .map(|_| ())
        .map_err(|error| ToolError::InvalidArguments(error.to_string()))
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn read_limited(path: &Path, limit: usize) -> Result<(String, bool), ToolError> {
    let bytes = std::fs::read(path).map_err(|error| ToolError::Io(error.to_string()))?;
    if bytes.iter().take(8192).any(|byte| *byte == 0) {
        return Err(ToolError::Io("binary file is not supported".to_owned()));
    }
    let truncated = bytes.len() > limit;
    let slice = &bytes[..bytes.len().min(limit)];
    let text = std::str::from_utf8(slice)
        .map_err(|error| ToolError::Io(format!("file is not UTF-8: {error}")))?;
    Ok((text.to_owned(), truncated))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PathArg {
    path: String,
}

struct ListDirectory;
#[async_trait]
impl Tool for ListDirectory {
    fn definition(&self) -> ToolDefinition {
        schema(
            "list_directory",
            "List entries in a workspace directory",
            json!({"path":{"type":"string"}}),
            &["path"],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Read)
    }
    fn validate(&self, v: &Value) -> Result<(), ToolError> {
        validate_decode::<PathArg>(v)
    }
    async fn execute(&self, ctx: &ToolContext, v: Value) -> Result<ToolResult, ToolError> {
        let args: PathArg = decode(v)?;
        let path = ctx.workspace.resolve_existing(args.path)?;
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(&path).map_err(|e| ToolError::Io(e.to_string()))? {
            let entry = entry.map_err(|e| ToolError::Io(e.to_string()))?;
            let kind = entry
                .file_type()
                .map_err(|e| ToolError::Io(e.to_string()))?;
            entries.push(format!(
                "{}\t{}",
                if kind.is_dir() {
                    "dir"
                } else if kind.is_symlink() {
                    "symlink"
                } else {
                    "file"
                },
                entry.file_name().to_string_lossy()
            ));
        }
        entries.sort();
        let truncated = entries.len() > ctx.limits.search_result_limit;
        entries.truncate(ctx.limits.search_result_limit);
        Ok(ToolResult::text(
            entries.join("\n"),
            format!("listed {}", relative(ctx.workspace.root(), &path)),
            truncated,
        ))
    }
}

struct ReadFile;
#[async_trait]
impl Tool for ReadFile {
    fn definition(&self) -> ToolDefinition {
        schema(
            "read_file",
            "Read a UTF-8 workspace file",
            json!({"path":{"type":"string"}}),
            &["path"],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Read)
    }
    fn validate(&self, v: &Value) -> Result<(), ToolError> {
        validate_decode::<PathArg>(v)
    }
    async fn execute(&self, ctx: &ToolContext, v: Value) -> Result<ToolResult, ToolError> {
        let args: PathArg = decode(v)?;
        let path = ctx.workspace.resolve_existing(args.path)?;
        let (text, truncated) = read_limited(&path, ctx.limits.file_read_limit_bytes)?;
        Ok(ToolResult::text(
            text,
            format!("read {}", relative(ctx.workspace.root(), &path)),
            truncated,
        ))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RangeArg {
    path: String,
    start_line: usize,
    end_line: usize,
}
struct ReadFileRange;
#[async_trait]
impl Tool for ReadFileRange {
    fn definition(&self) -> ToolDefinition {
        schema(
            "read_file_range",
            "Read an inclusive line range",
            json!({
                "path":{"type":"string"}, "start_line":{"type":"integer","minimum":1}, "end_line":{"type":"integer","minimum":1}
            }),
            &["path", "start_line", "end_line"],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Read)
    }
    fn validate(&self, v: &Value) -> Result<(), ToolError> {
        let a: RangeArg = serde_json::from_value(v.clone())
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        if a.start_line == 0 || a.end_line < a.start_line || a.end_line - a.start_line > 2000 {
            return Err(ToolError::InvalidArguments(
                "invalid or excessive line range".to_owned(),
            ));
        }
        Ok(())
    }
    async fn execute(&self, ctx: &ToolContext, v: Value) -> Result<ToolResult, ToolError> {
        let a: RangeArg = decode(v)?;
        let path = ctx.workspace.resolve_existing(a.path)?;
        let (text, source_truncated) = read_limited(&path, ctx.limits.file_read_limit_bytes)?;
        let lines: Vec<_> = text
            .lines()
            .enumerate()
            .filter(|(i, _)| (a.start_line..=a.end_line).contains(&(i + 1)))
            .map(|(i, line)| format!("{}: {}", i + 1, line))
            .collect();
        Ok(ToolResult::text(
            lines.join("\n"),
            format!("read lines {}-{}", a.start_line, a.end_line),
            source_truncated,
        ))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GlobArg {
    pattern: String,
}
struct GlobFiles;
#[async_trait]
impl Tool for GlobFiles {
    fn definition(&self) -> ToolDefinition {
        schema(
            "glob",
            "Find workspace paths matching a glob",
            json!({"pattern":{"type":"string"}}),
            &["pattern"],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Read)
    }
    fn validate(&self, v: &Value) -> Result<(), ToolError> {
        let a: GlobArg = serde_json::from_value(v.clone())
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        Glob::new(&a.pattern)
            .map(|_| ())
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))
    }
    async fn execute(&self, ctx: &ToolContext, v: Value) -> Result<ToolResult, ToolError> {
        let a: GlobArg = decode(v)?;
        let mut builder = GlobSetBuilder::new();
        builder.add(Glob::new(&a.pattern).map_err(|e| ToolError::InvalidArguments(e.to_string()))?);
        let set = builder
            .build()
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        let mut found = Vec::new();
        let mut truncated = false;
        for entry in WalkDir::new(ctx.workspace.root())
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
        {
            let rel = entry
                .path()
                .strip_prefix(ctx.workspace.root())
                .unwrap_or(entry.path());
            if set.is_match(rel) {
                found.push(relative(ctx.workspace.root(), entry.path()));
                if found.len() >= ctx.limits.search_result_limit {
                    truncated = true;
                    break;
                }
            }
        }
        found.sort();
        Ok(ToolResult::text(
            found.join("\n"),
            format!("matched {} paths", found.len()),
            truncated,
        ))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GrepArg {
    pattern: String,
    path: Option<String>,
}
struct Grep;
#[async_trait]
impl Tool for Grep {
    fn definition(&self) -> ToolDefinition {
        schema(
            "grep",
            "Search UTF-8 workspace files with a regular expression",
            json!({
                "pattern":{"type":"string"}, "path":{"type":"string"}
            }),
            &["pattern"],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Read)
    }
    fn validate(&self, v: &Value) -> Result<(), ToolError> {
        let a: GrepArg = serde_json::from_value(v.clone())
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        Regex::new(&a.pattern)
            .map(|_| ())
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))
    }
    async fn execute(&self, ctx: &ToolContext, v: Value) -> Result<ToolResult, ToolError> {
        let a: GrepArg = decode(v)?;
        let regex =
            Regex::new(&a.pattern).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        let root = match a.path {
            Some(path) => ctx.workspace.resolve_existing(path)?,
            None => ctx.workspace.root().to_path_buf(),
        };
        let mut matches = Vec::new();
        let mut truncated = false;
        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
        {
            let Ok((text, _)) = read_limited(entry.path(), ctx.limits.file_read_limit_bytes) else {
                continue;
            };
            for (index, line) in text.lines().enumerate() {
                if regex.is_match(line) {
                    matches.push(format!(
                        "{}:{}:{}",
                        relative(ctx.workspace.root(), entry.path()),
                        index + 1,
                        line
                    ));
                    if matches.len() >= ctx.limits.search_result_limit {
                        truncated = true;
                        break;
                    }
                }
            }
            if truncated {
                break;
            }
        }
        Ok(ToolResult::text(
            matches.join("\n"),
            format!("found {} matches", matches.len()),
            truncated,
        ))
    }
}

struct GitStatus;
struct GitDiff;
macro_rules! git_tool {
    ($ty:ident, $name:literal, $desc:literal, $args:expr) => {
        #[async_trait]
        impl Tool for $ty {
            fn definition(&self) -> ToolDefinition {
                schema($name, $desc, json!({}), &[])
            }
            fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
                Ok(RiskLevel::Read)
            }
            fn validate(&self, v: &Value) -> Result<(), ToolError> {
                if v.as_object().is_some_and(serde_json::Map::is_empty) {
                    Ok(())
                } else {
                    Err(ToolError::InvalidArguments(
                        "expected an empty object".to_owned(),
                    ))
                }
            }
            async fn execute(&self, ctx: &ToolContext, _: Value) -> Result<ToolResult, ToolError> {
                fixed_command(ctx, "git", $args, $name).await
            }
        }
    };
}
git_tool!(
    GitStatus,
    "git_status",
    "Show read-only Git status",
    &["status", "--short"]
);
git_tool!(
    GitDiff,
    "git_diff",
    "Show unstaged Git diff",
    &["diff", "--no-ext-diff", "--"]
);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PatchArg {
    path: String,
    patch: String,
    expected_sha256: Option<String>,
}
struct PatchFile;
#[async_trait]
impl Tool for PatchFile {
    fn definition(&self) -> ToolDefinition {
        schema(
            "patch_file",
            "Apply a unified patch to an existing UTF-8 file",
            json!({
                "path":{"type":"string"}, "patch":{"type":"string"}, "expected_sha256":{"type":"string"}
            }),
            &["path", "patch"],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Modify)
    }
    fn validate(&self, v: &Value) -> Result<(), ToolError> {
        let a: PatchArg = serde_json::from_value(v.clone())
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        diffy::Patch::from_str(&a.patch)
            .map(|_| ())
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))
    }
    async fn execute(&self, ctx: &ToolContext, v: Value) -> Result<ToolResult, ToolError> {
        let a: PatchArg = decode(v)?;
        let path = ctx.workspace.resolve_existing(&a.path)?;
        let original = std::fs::read_to_string(&path).map_err(|e| ToolError::Io(e.to_string()))?;
        if let Some(expected) = a.expected_sha256 {
            if sha256(original.as_bytes()) != expected {
                return Err(ToolError::Conflict(a.path));
            }
        }
        let patch = diffy::Patch::from_str(&a.patch)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        let updated =
            diffy::apply(&original, &patch).map_err(|e| ToolError::Conflict(e.to_string()))?;
        if updated.len() > ctx.limits.file_read_limit_bytes * 4 {
            return Err(ToolError::Policy("resulting file is too large".to_owned()));
        }
        let checked = ctx.workspace.resolve_existing(&path)?;
        if checked != path {
            return Err(ToolError::Conflict(a.path));
        }
        std::fs::write(&path, updated.as_bytes()).map_err(|e| ToolError::Io(e.to_string()))?;
        Ok(ToolResult {
            content: json!({"path":relative(ctx.workspace.root(), &path),"sha256":sha256(updated.as_bytes())}),
            summary: format!("patched {}", a.path),
            truncated: false,
            metadata: json!({"bytes":updated.len()}),
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteArg {
    path: String,
    content: String,
    #[serde(default)]
    overwrite: bool,
    expected_sha256: Option<String>,
}
struct WriteFile;
#[async_trait]
impl Tool for WriteFile {
    fn definition(&self) -> ToolDefinition {
        schema(
            "write_file",
            "Create or explicitly overwrite a UTF-8 file",
            json!({
                "path":{"type":"string"}, "content":{"type":"string"}, "overwrite":{"type":"boolean"}, "expected_sha256":{"type":"string"}
            }),
            &["path", "content"],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Modify)
    }
    fn validate(&self, v: &Value) -> Result<(), ToolError> {
        validate_decode::<WriteArg>(v)
    }
    async fn execute(&self, ctx: &ToolContext, v: Value) -> Result<ToolResult, ToolError> {
        let a: WriteArg = decode(v)?;
        let path = ctx.workspace.resolve_new(&a.path)?;
        if a.content.len() > ctx.limits.file_read_limit_bytes * 4 {
            return Err(ToolError::Policy("content is too large".to_owned()));
        }
        if path.exists() {
            if !a.overwrite {
                return Err(ToolError::Conflict("overwrite=true is required".to_owned()));
            }
            if let Some(expected) = &a.expected_sha256 {
                let current = std::fs::read(&path).map_err(|e| ToolError::Io(e.to_string()))?;
                if sha256(&current) != *expected {
                    return Err(ToolError::Conflict(a.path));
                }
            }
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ToolError::Io(e.to_string()))?;
        }
        let checked = ctx.workspace.resolve_new(&path)?;
        if checked != path {
            return Err(ToolError::Conflict(a.path));
        }
        std::fs::write(&path, a.content.as_bytes()).map_err(|e| ToolError::Io(e.to_string()))?;
        Ok(ToolResult {
            content: json!({"path":relative(ctx.workspace.root(), &path),"sha256":sha256(a.content.as_bytes())}),
            summary: format!("wrote {}", a.path),
            truncated: false,
            metadata: json!({"bytes":a.content.len(),"overwrite":a.overwrite}),
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandArg {
    program: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default = "dot")]
    working_directory: String,
    timeout_seconds: Option<u64>,
    #[serde(default)]
    environment: BTreeMap<String, String>,
}
fn dot() -> String {
    ".".to_owned()
}
struct RunCommand;
#[async_trait]
impl Tool for RunCommand {
    fn definition(&self) -> ToolDefinition {
        schema(
            "run_command",
            "Run a structured non-interactive command",
            json!({
                "program":{"type":"string"}, "args":{"type":"array","items":{"type":"string"}}, "working_directory":{"type":"string"},
                "timeout_seconds":{"type":"integer","minimum":1}, "environment":{"type":"object","additionalProperties":{"type":"string"}}
            }),
            &["program"],
        )
    }
    fn risk(&self, v: &Value) -> Result<RiskLevel, ToolError> {
        let a: CommandArg = serde_json::from_value(v.clone())
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        Ok(command_risk(&a))
    }
    fn validate(&self, v: &Value) -> Result<(), ToolError> {
        let a: CommandArg = serde_json::from_value(v.clone())
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        if a.program.trim().is_empty() {
            Err(ToolError::InvalidArguments("program is empty".to_owned()))
        } else if a.timeout_seconds == Some(0) {
            Err(ToolError::InvalidArguments(
                "timeout must be positive".to_owned(),
            ))
        } else {
            Ok(())
        }
    }
    async fn execute(&self, ctx: &ToolContext, v: Value) -> Result<ToolResult, ToolError> {
        let a: CommandArg = decode(v)?;
        let cwd = ctx.workspace.resolve_existing(&a.working_directory)?;
        let timeout = a
            .timeout_seconds
            .unwrap_or(ctx.limits.command_timeout_seconds)
            .min(ctx.limits.command_timeout_seconds);
        run_process(ctx, &a.program, &a.args, &cwd, &a.environment, timeout).await
    }
}

fn command_risk(a: &CommandArg) -> RiskLevel {
    let program = Path::new(&a.program)
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or(&a.program)
        .to_ascii_lowercase();
    if ["rm", "sudo", "chmod", "chown", "del", "erase", "rmdir"].contains(&program.as_str()) {
        return RiskLevel::Dangerous;
    }
    if program == "git"
        && a.args
            .iter()
            .any(|arg| ["push", "reset", "clean"].contains(&arg.as_str()))
    {
        return RiskLevel::Dangerous;
    }
    if ["curl", "wget"].contains(&program.as_str())
        && a.args.iter().any(|arg| {
            ["-x", "--request", "--upload-file", "-t"].contains(&arg.to_ascii_lowercase().as_str())
        })
    {
        return RiskLevel::Dangerous;
    }
    RiskLevel::Execute
}

async fn fixed_command(
    ctx: &ToolContext,
    program: &str,
    args: &[&str],
    name: &str,
) -> Result<ToolResult, ToolError> {
    let args: Vec<String> = args.iter().map(|v| (*v).to_owned()).collect();
    run_process(
        ctx,
        program,
        &args,
        ctx.workspace.root(),
        &BTreeMap::new(),
        ctx.limits.command_timeout_seconds,
    )
    .await
    .map(|mut result| {
        result.summary = name.to_owned();
        result
    })
}

async fn run_process(
    ctx: &ToolContext,
    program: &str,
    args: &[String],
    cwd: &Path,
    env: &BTreeMap<String, String>,
    timeout_seconds: u64,
) -> Result<ToolResult, ToolError> {
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(cwd)
        .envs(env)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command
        .spawn()
        .map_err(|e| ToolError::Execution(e.to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ToolError::Execution("stdout unavailable".to_owned()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| ToolError::Execution("stderr unavailable".to_owned()))?;
    let stdout_task = tokio::spawn(read_stream_limited(stdout, ctx.limits.stdout_limit_bytes));
    let stderr_task = tokio::spawn(read_stream_limited(stderr, ctx.limits.stderr_limit_bytes));
    let started = Instant::now();
    let status = tokio::select! {
        result = child.wait() => result.map_err(|e| ToolError::Execution(e.to_string()))?,
        () = tokio::time::sleep(Duration::from_secs(timeout_seconds)) => { let _ = child.kill().await; let _ = child.wait().await; return Err(ToolError::Timeout(timeout_seconds)); },
        () = ctx.cancellation.cancelled() => { let _ = child.kill().await; let _ = child.wait().await; return Err(ToolError::Cancelled); },
    };
    let (stdout, stdout_truncated) = stdout_task
        .await
        .map_err(|e| ToolError::Execution(e.to_string()))??;
    let (stderr, stderr_truncated) = stderr_task
        .await
        .map_err(|e| ToolError::Execution(e.to_string()))??;
    let success = status.success();
    Ok(ToolResult {
        content: json!({"exit_code":status.code(),"stdout":stdout,"stderr":stderr}),
        summary: format!(
            "command {} ({})",
            if success { "succeeded" } else { "failed" },
            status
        ),
        truncated: stdout_truncated || stderr_truncated,
        metadata: json!({"duration_ms":started.elapsed().as_millis(),"success":success}),
    })
}

async fn read_stream_limited<R: AsyncRead + Unpin>(
    mut reader: R,
    limit: usize,
) -> Result<(String, bool), ToolError> {
    let mut kept = Vec::new();
    let mut buffer = [0_u8; 8192];
    let mut truncated = false;
    loop {
        let count = reader
            .read(&mut buffer)
            .await
            .map_err(|e| ToolError::Io(e.to_string()))?;
        if count == 0 {
            break;
        }
        let available = limit.saturating_sub(kept.len());
        let take = count.min(available);
        kept.extend_from_slice(&buffer[..take]);
        if take < count {
            truncated = true;
        }
    }
    Ok((String::from_utf8_lossy(&kept).into_owned(), truncated))
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(root: &Path) -> Result<ToolContext, Box<dyn std::error::Error>> {
        Ok(ToolContext {
            session_id: agent_security::SessionId::new(),
            task_id: agent_security::TaskId::new(),
            call_id: agent_security::ToolCallId::new(),
            workspace: agent_security::WorkspaceGuard::new(root)?,
            cancellation: tokio_util::sync::CancellationToken::new(),
            limits: crate::ExecutionLimits {
                stdout_limit_bytes: 5,
                stderr_limit_bytes: 5,
                command_timeout_seconds: 1,
                ..crate::ExecutionLimits::default()
            },
        })
    }

    #[test]
    fn dangerous_commands_are_escalated() {
        let a = CommandArg {
            program: "git".to_owned(),
            args: vec!["reset".to_owned(), "--hard".to_owned()],
            working_directory: ".".to_owned(),
            timeout_seconds: None,
            environment: BTreeMap::new(),
        };
        assert_eq!(command_risk(&a), RiskLevel::Dangerous);
    }
    #[test]
    fn write_requires_explicit_overwrite() {
        let value = json!({"path":"a","content":"x"});
        assert!(validate_decode::<WriteArg>(&value).is_ok());
    }

    #[tokio::test]
    async fn write_and_patch_enforce_conflicts() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let ctx = context(temp.path())?;
        WriteFile
            .execute(&ctx, json!({"path":"a.txt","content":"old\n"}))
            .await?;
        assert!(
            WriteFile
                .execute(&ctx, json!({"path":"a.txt","content":"bad\n"}))
                .await
                .is_err()
        );
        let original = std::fs::read(temp.path().join("a.txt"))?;
        PatchFile
            .execute(
                &ctx,
                json!({
                    "path":"a.txt",
                    "expected_sha256":sha256(&original),
                    "patch":"--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n"
                }),
            )
            .await?;
        assert_eq!(std::fs::read_to_string(temp.path().join("a.txt"))?, "new\n");
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn command_output_is_truncated_and_timeout_is_reported()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let ctx = context(temp.path())?;
        let result = RunCommand
            .execute(
                &ctx,
                json!({"program":"sh","args":["-c","printf 123456789"]}),
            )
            .await?;
        assert!(result.truncated);
        assert_eq!(result.content["stdout"], "12345");
        let timeout = RunCommand
            .execute(
                &ctx,
                json!({"program":"sh","args":["-c","sleep 2"],"timeout_seconds":1}),
            )
            .await;
        assert!(matches!(timeout, Err(ToolError::Timeout(1))));
        Ok(())
    }
}
