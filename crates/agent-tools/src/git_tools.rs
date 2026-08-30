use crate::process::run_process;
use crate::{CommandProfile, Tool, ToolContext, ToolError, ToolRegistry, ToolResult};
use agent_model::ToolDefinition;
use agent_security::RiskLevel;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;
use uuid::Uuid;

pub(crate) fn register(registry: &mut ToolRegistry) -> Result<(), ToolError> {
    registry.register(GitStatus)?;
    registry.register(GitDiff)?;
    registry.register(GitLog)?;
    registry.register(GitShow)?;
    registry.register(GitBranch)?;
    registry.register(GitCheckout)?;
    registry.register(GitCommit)?;
    registry.register(GitCheckpoint)?;
    Ok(())
}

fn schema(name: &str, description: &str, properties: Value, required: &[&str]) -> ToolDefinition {
    ToolDefinition::function(
        name,
        description,
        json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}),
    )
}

fn decode<T: for<'de> Deserialize<'de>>(value: Value) -> Result<T, ToolError> {
    serde_json::from_value(value).map_err(|error| ToolError::InvalidArguments(error.to_string()))
}

fn validate_empty(value: &Value) -> Result<(), ToolError> {
    if value.as_object().is_some_and(serde_json::Map::is_empty) {
        Ok(())
    } else {
        Err(ToolError::InvalidArguments(
            "expected an empty object".to_owned(),
        ))
    }
}

fn validate_token(value: &str, kind: &str) -> Result<(), ToolError> {
    if value.is_empty()
        || value.starts_with('-')
        || value.contains(char::is_whitespace)
        || value
            .chars()
            .any(|character| "~^:?*[\\".contains(character))
        || value.contains("..")
    {
        return Err(ToolError::InvalidArguments(format!(
            "unsafe {kind}: {value}"
        )));
    }
    Ok(())
}

async fn git(context: &ToolContext, args: Vec<String>) -> Result<ToolResult, ToolError> {
    let limits = context.limits.command_limits(CommandProfile::Git);
    run_process(
        context,
        "git",
        &args,
        context.workspace.root(),
        &BTreeMap::new(),
        &limits,
    )
    .await
}

fn succeeded(result: &ToolResult) -> bool {
    result.metadata["success"].as_bool() == Some(true)
}

fn stdout(result: &ToolResult) -> &str {
    result.content["stdout"].as_str().unwrap_or_default()
}

struct GitStatus;
#[async_trait]
impl Tool for GitStatus {
    fn definition(&self) -> ToolDefinition {
        schema("git_status", "Show Git status", json!({}), &[])
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Read)
    }
    fn validate(&self, value: &Value) -> Result<(), ToolError> {
        validate_empty(value)
    }
    async fn execute(&self, context: &ToolContext, _: Value) -> Result<ToolResult, ToolError> {
        let mut result = git(context, vec!["status".to_owned(), "--short".to_owned()]).await?;
        result.summary = "git status".to_owned();
        result.metadata["kind"] = json!("git_status");
        Ok(result)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DiffArg {
    #[serde(default)]
    staged: bool,
    base: Option<String>,
    path: Option<String>,
}

struct GitDiff;
#[async_trait]
impl Tool for GitDiff {
    fn definition(&self) -> ToolDefinition {
        schema(
            "git_diff",
            "Show a bounded Git diff for final review",
            json!({"staged":{"type":"boolean"},"base":{"type":"string"},"path":{"type":"string"}}),
            &[],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Read)
    }
    fn validate(&self, value: &Value) -> Result<(), ToolError> {
        let args: DiffArg = serde_json::from_value(value.clone())
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        if args.staged && args.base.is_some() {
            return Err(ToolError::InvalidArguments(
                "staged and base are mutually exclusive".to_owned(),
            ));
        }
        if let Some(base) = &args.base {
            validate_token(base, "revision")?;
        }
        Ok(())
    }
    async fn execute(&self, context: &ToolContext, value: Value) -> Result<ToolResult, ToolError> {
        let args: DiffArg = decode(value)?;
        let include_untracked = !args.staged && args.base.is_none();
        let scoped_path = match &args.path {
            Some(path) => {
                let resolved = context.workspace.resolve_new(path)?;
                Some(relative(context.workspace.root(), &resolved))
            }
            None => None,
        };
        let mut command = vec!["diff".to_owned(), "--no-ext-diff".to_owned()];
        if args.staged {
            command.push("--cached".to_owned());
        } else if let Some(base) = &args.base {
            command.push(base.clone());
        }
        command.push("--".to_owned());
        if let Some(path) = &scoped_path {
            command.push(path.clone());
        }
        let mut result = git(context, command).await?;
        let not_repository = !succeeded(&result)
            && result.content["stderr"]
                .as_str()
                .is_some_and(|stderr| stderr.to_ascii_lowercase().contains("not a git repository"));
        if not_repository {
            result.summary = "git review unavailable: workspace is not a repository".to_owned();
            result.metadata["success"] = json!(true);
            result.metadata["review_unavailable"] = json!(true);
        } else {
            result.summary = "git diff reviewed".to_owned();
            if succeeded(&result) && include_untracked {
                append_untracked_review(context, &mut result, scoped_path.as_deref()).await?;
            }
        }
        result.metadata["kind"] = json!("git_diff");
        Ok(result)
    }
}

async fn append_untracked_review(
    context: &ToolContext,
    result: &mut ToolResult,
    path: Option<&str>,
) -> Result<(), ToolError> {
    let mut command = vec![
        "ls-files".to_owned(),
        "--others".to_owned(),
        "--exclude-standard".to_owned(),
        "-z".to_owned(),
        "--".to_owned(),
    ];
    if let Some(path) = path {
        command.push(path.to_owned());
    }
    let listed = git(context, command).await?;
    if !succeeded(&listed) {
        return Ok(());
    }
    let paths: Vec<String> = stdout(&listed)
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect();
    if paths.is_empty() {
        result.metadata["untracked_files"] = json!([]);
        return Ok(());
    }

    let mut combined = stdout(result).to_owned();
    let limit = context
        .limits
        .command_limits(CommandProfile::Git)
        .stdout_limit_bytes;
    let mut truncated = false;
    for path in &paths {
        let resolved = context.workspace.resolve_existing(path)?;
        let bytes = std::fs::read(&resolved).map_err(|error| ToolError::Io(error.to_string()))?;
        let patch = if bytes.iter().take(8192).any(|byte| *byte == 0) {
            format!("\ndiff --git a/{path} b/{path}\nnew untracked binary file\n")
        } else {
            let text = String::from_utf8_lossy(&bytes);
            let lines = text.lines().count();
            let mut additions = String::new();
            for line in text.split_inclusive('\n') {
                additions.push('+');
                additions.push_str(line);
            }
            format!(
                "\ndiff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{lines} @@\n{additions}"
            )
        };
        truncated |= append_bounded(&mut combined, &patch, limit);
    }
    result.content["stdout"] = json!(combined);
    result.metadata["untracked_files"] = json!(paths);
    result.metadata["untracked_included"] = json!(true);
    result.truncated |= truncated || listed.truncated;
    result.summary = format!(
        "git diff reviewed including {} untracked files",
        paths.len()
    );
    Ok(())
}

fn append_bounded(output: &mut String, addition: &str, limit: usize) -> bool {
    let available = limit.saturating_sub(output.len());
    if addition.len() <= available {
        output.push_str(addition);
        return false;
    }
    if available == 0 {
        return true;
    }
    const MARKER: &[u8] = b"\n... untracked diff truncated ...\n";
    let bytes = addition.as_bytes();
    if available <= MARKER.len() {
        output.push_str(&String::from_utf8_lossy(
            &bytes[..available.min(bytes.len())],
        ));
        return true;
    }
    let payload = available - MARKER.len();
    let head = payload / 2;
    let tail = payload - head;
    let mut kept = Vec::with_capacity(available);
    kept.extend_from_slice(&bytes[..head.min(bytes.len())]);
    kept.extend_from_slice(MARKER);
    kept.extend_from_slice(&bytes[bytes.len().saturating_sub(tail)..]);
    output.push_str(&String::from_utf8_lossy(&kept));
    true
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LogArg {
    max_count: Option<usize>,
    path: Option<String>,
}
struct GitLog;
#[async_trait]
impl Tool for GitLog {
    fn definition(&self) -> ToolDefinition {
        schema(
            "git_log",
            "Show recent Git commits",
            json!({"max_count":{"type":"integer","minimum":1,"maximum":100},"path":{"type":"string"}}),
            &[],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Read)
    }
    fn validate(&self, value: &Value) -> Result<(), ToolError> {
        let args: LogArg = serde_json::from_value(value.clone())
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        if args
            .max_count
            .is_some_and(|count| count == 0 || count > 100)
        {
            return Err(ToolError::InvalidArguments(
                "max_count must be between 1 and 100".to_owned(),
            ));
        }
        Ok(())
    }
    async fn execute(&self, context: &ToolContext, value: Value) -> Result<ToolResult, ToolError> {
        let args: LogArg = decode(value)?;
        let mut command = vec![
            "log".to_owned(),
            format!("--max-count={}", args.max_count.unwrap_or(20)),
            "--oneline".to_owned(),
            "--decorate".to_owned(),
        ];
        if let Some(path) = args.path {
            let resolved = context.workspace.resolve_new(&path)?;
            command.extend([
                "--".to_owned(),
                relative(context.workspace.root(), &resolved),
            ]);
        }
        let mut result = git(context, command).await?;
        result.summary = "git log".to_owned();
        result.metadata["kind"] = json!("git_log");
        Ok(result)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShowArg {
    revision: String,
    path: Option<String>,
}
struct GitShow;
#[async_trait]
impl Tool for GitShow {
    fn definition(&self) -> ToolDefinition {
        schema(
            "git_show",
            "Show one Git revision",
            json!({"revision":{"type":"string"},"path":{"type":"string"}}),
            &["revision"],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Read)
    }
    fn validate(&self, value: &Value) -> Result<(), ToolError> {
        let args: ShowArg = serde_json::from_value(value.clone())
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        validate_token(&args.revision, "revision")
    }
    async fn execute(&self, context: &ToolContext, value: Value) -> Result<ToolResult, ToolError> {
        let args: ShowArg = decode(value)?;
        let mut command = vec!["show".to_owned(), "--no-ext-diff".to_owned(), args.revision];
        if let Some(path) = args.path {
            let resolved = context.workspace.resolve_new(&path)?;
            command.extend([
                "--".to_owned(),
                relative(context.workspace.root(), &resolved),
            ]);
        }
        let mut result = git(context, command).await?;
        result.summary = "git show".to_owned();
        result.metadata["kind"] = json!("git_show");
        Ok(result)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum BranchAction {
    List,
    Create,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BranchArg {
    action: BranchAction,
    name: Option<String>,
}
struct GitBranch;
#[async_trait]
impl Tool for GitBranch {
    fn definition(&self) -> ToolDefinition {
        schema(
            "git_branch",
            "List or safely create a Git branch",
            json!({"action":{"type":"string","enum":["list","create"]},"name":{"type":"string"}}),
            &["action"],
        )
    }
    fn risk(&self, value: &Value) -> Result<RiskLevel, ToolError> {
        let args: BranchArg = serde_json::from_value(value.clone())
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        Ok(if matches!(args.action, BranchAction::List) {
            RiskLevel::Read
        } else {
            RiskLevel::Modify
        })
    }
    fn validate(&self, value: &Value) -> Result<(), ToolError> {
        let args: BranchArg = serde_json::from_value(value.clone())
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        match args.action {
            BranchAction::List if args.name.is_none() => Ok(()),
            BranchAction::Create => args
                .name
                .as_deref()
                .ok_or_else(|| {
                    ToolError::InvalidArguments("name is required for create".to_owned())
                })
                .and_then(|name| validate_token(name, "branch name")),
            BranchAction::List => Err(ToolError::InvalidArguments(
                "name is not valid for list".to_owned(),
            )),
        }
    }
    async fn execute(&self, context: &ToolContext, value: Value) -> Result<ToolResult, ToolError> {
        let args: BranchArg = decode(value)?;
        let command = match args.action {
            BranchAction::List => vec!["branch".to_owned(), "--list".to_owned()],
            BranchAction::Create => vec![
                "branch".to_owned(),
                "--".to_owned(),
                args.name.unwrap_or_default(),
            ],
        };
        let mut result = git(context, command).await?;
        result.summary = "git branch".to_owned();
        result.metadata["kind"] = json!("git_branch");
        Ok(result)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckoutArg {
    branch: String,
    #[serde(default)]
    create: bool,
}
struct GitCheckout;
#[async_trait]
impl Tool for GitCheckout {
    fn definition(&self) -> ToolDefinition {
        schema(
            "git_checkout",
            "Switch to a branch only when the worktree is clean",
            json!({"branch":{"type":"string"},"create":{"type":"boolean"}}),
            &["branch"],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Modify)
    }
    fn validate(&self, value: &Value) -> Result<(), ToolError> {
        let args: CheckoutArg = serde_json::from_value(value.clone())
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        validate_token(&args.branch, "branch name")
    }
    async fn execute(&self, context: &ToolContext, value: Value) -> Result<ToolResult, ToolError> {
        let args: CheckoutArg = decode(value)?;
        let status = git(
            context,
            vec![
                "status".to_owned(),
                "--porcelain".to_owned(),
                "--untracked-files=normal".to_owned(),
            ],
        )
        .await?;
        if !succeeded(&status) {
            return Ok(status);
        }
        if !stdout(&status).trim().is_empty() {
            return Err(ToolError::Conflict(
                "branch switch requires a clean worktree".to_owned(),
            ));
        }
        let mut command = vec!["switch".to_owned()];
        if args.create {
            command.push("-c".to_owned());
        }
        command.extend(["--".to_owned(), args.branch]);
        let mut result = git(context, command).await?;
        result.summary = "git branch switched".to_owned();
        result.metadata["kind"] = json!("git_checkout");
        Ok(result)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CommitArg {
    message: String,
    paths: Vec<String>,
}
struct GitCommit;
#[async_trait]
impl Tool for GitCommit {
    fn definition(&self) -> ToolDefinition {
        schema(
            "git_commit",
            "Commit only explicitly selected paths without running hooks",
            json!({"message":{"type":"string"},"paths":{"type":"array","items":{"type":"string"},"minItems":1}}),
            &["message", "paths"],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Modify)
    }
    fn validate(&self, value: &Value) -> Result<(), ToolError> {
        let args: CommitArg = serde_json::from_value(value.clone())
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        if args.message.trim().is_empty() || args.paths.is_empty() {
            return Err(ToolError::InvalidArguments(
                "non-empty message and paths are required".to_owned(),
            ));
        }
        Ok(())
    }
    async fn execute(&self, context: &ToolContext, value: Value) -> Result<ToolResult, ToolError> {
        let args: CommitArg = decode(value)?;
        let hooks = tempfile::tempdir().map_err(|error| ToolError::Io(error.to_string()))?;
        let mut command = vec![
            "-c".to_owned(),
            format!("core.hooksPath={}", hooks.path().display()),
            "commit".to_owned(),
            "--only".to_owned(),
            "--no-verify".to_owned(),
            "--message".to_owned(),
            args.message,
            "--".to_owned(),
        ];
        for path in args.paths {
            let resolved = context.workspace.resolve_new(&path)?;
            command.push(relative(context.workspace.root(), &resolved));
        }
        let mut result = git(context, command).await?;
        result.summary = "git commit".to_owned();
        result.metadata["kind"] = json!("git_commit");
        Ok(result)
    }
}

struct GitCheckpoint;
#[async_trait]
impl Tool for GitCheckpoint {
    fn definition(&self) -> ToolDefinition {
        schema(
            "git_checkpoint",
            "Save tracked changes under refs/veyra/checkpoints without changing the worktree",
            json!({}),
            &[],
        )
    }
    fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
        Ok(RiskLevel::Modify)
    }
    fn validate(&self, value: &Value) -> Result<(), ToolError> {
        validate_empty(value)
    }
    async fn execute(&self, context: &ToolContext, _: Value) -> Result<ToolResult, ToolError> {
        let status = git(
            context,
            vec![
                "status".to_owned(),
                "--porcelain".to_owned(),
                "--untracked-files=normal".to_owned(),
            ],
        )
        .await?;
        if !succeeded(&status) {
            return Ok(status);
        }
        let untracked: Vec<&str> = stdout(&status)
            .lines()
            .filter(|line| line.starts_with("?? "))
            .collect();
        let snapshot = git(
            context,
            vec![
                "stash".to_owned(),
                "create".to_owned(),
                "Veyra checkpoint".to_owned(),
            ],
        )
        .await?;
        if !succeeded(&snapshot) {
            return Ok(snapshot);
        }
        let object = stdout(&snapshot).trim();
        if object.is_empty() {
            return Err(ToolError::Conflict(
                "no tracked changes to checkpoint".to_owned(),
            ));
        }
        validate_token(object, "object id")?;
        let reference = format!("refs/veyra/checkpoints/{}", Uuid::new_v4());
        let updated = git(
            context,
            vec![
                "update-ref".to_owned(),
                reference.clone(),
                object.to_owned(),
            ],
        )
        .await?;
        if !succeeded(&updated) {
            return Ok(updated);
        }
        Ok(ToolResult {
            content: json!({"reference":reference,"object":object,"untracked_excluded":untracked}),
            summary: "git checkpoint created without changing the worktree".to_owned(),
            truncated: false,
            metadata: json!({"kind":"git_checkpoint","success":true,"untracked_excluded_count":untracked.len()}),
        })
    }
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExecutionLimits;
    use std::fs;
    use std::process::Command;
    use tokio_util::sync::CancellationToken;

    fn command(root: &Path, args: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
        let output = Command::new("git").args(args).current_dir(root).output()?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned().into());
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    fn repository() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        command(root.path(), &["init", "-q"])?;
        command(root.path(), &["config", "user.name", "Veyra Test"])?;
        command(
            root.path(),
            &["config", "user.email", "veyra@example.invalid"],
        )?;
        fs::write(root.path().join("a.txt"), "a0\n")?;
        fs::write(root.path().join("b.txt"), "b0\n")?;
        command(root.path(), &["add", "a.txt", "b.txt"])?;
        command(root.path(), &["commit", "-q", "-m", "initial"])?;
        Ok(root)
    }

    fn context(root: &Path) -> Result<ToolContext, Box<dyn std::error::Error>> {
        Ok(ToolContext {
            session_id: agent_security::SessionId::new(),
            task_id: agent_security::TaskId::new(),
            call_id: agent_security::ToolCallId::new(),
            workspace: agent_security::WorkspaceGuard::new(root)?,
            cancellation: CancellationToken::new(),
            limits: ExecutionLimits::default(),
        })
    }

    #[test]
    fn rejects_option_injection() {
        assert!(validate_token("--all", "revision").is_err());
        assert!(GitShow.validate(&json!({"revision":"--stat"})).is_err());
    }

    #[tokio::test]
    async fn non_repository_diff_is_a_completed_unavailable_review()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let result = GitDiff.execute(&context(root.path())?, json!({})).await?;
        assert_eq!(result.metadata["success"], true);
        assert_eq!(result.metadata["review_unavailable"], true);
        Ok(())
    }

    #[tokio::test]
    async fn diff_review_includes_untracked_files_in_an_unborn_repository()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        command(root.path(), &["init", "-q"])?;
        fs::write(root.path().join("src.rs"), "fn answer() -> u32 { 42 }\n")?;
        let result = GitDiff.execute(&context(root.path())?, json!({})).await?;
        assert_eq!(result.metadata["success"], true);
        assert_eq!(result.metadata["untracked_included"], true);
        assert!(
            result.content["stdout"]
                .as_str()
                .is_some_and(|diff| diff.contains("+++ b/src.rs") && diff.contains("+fn answer()"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn checkpoint_preserves_worktree_and_reports_untracked()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = repository()?;
        fs::write(root.path().join("a.txt"), "changed\n")?;
        fs::write(root.path().join("untracked.txt"), "user data\n")?;
        command(root.path(), &["add", "a.txt"])?;
        let before = command(root.path(), &["status", "--porcelain=v1"])?;
        let result = GitCheckpoint
            .execute(&context(root.path())?, json!({}))
            .await?;
        let after = command(root.path(), &["status", "--porcelain=v1"])?;
        assert_eq!(before, after);
        assert_eq!(result.metadata["untracked_excluded_count"], 1);
        let reference = result.content["reference"].as_str().ok_or("missing ref")?;
        assert!(
            !command(root.path(), &["rev-parse", reference])?
                .trim()
                .is_empty()
        );
        Ok(())
    }

    #[tokio::test]
    async fn commit_is_path_scoped_and_checkout_rejects_dirty_tree()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = repository()?;
        fs::write(root.path().join("a.txt"), "a1\n")?;
        fs::write(root.path().join("b.txt"), "b1\n")?;
        command(root.path(), &["add", "b.txt"])?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let hook = root.path().join(".git/hooks/pre-commit");
            fs::write(&hook, "#!/bin/sh\ntouch hook-ran\n")?;
            let mut permissions = fs::metadata(&hook)?.permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&hook, permissions)?;
        }
        let result = GitCommit
            .execute(
                &context(root.path())?,
                json!({"message":"only a","paths":["a.txt"]}),
            )
            .await?;
        assert!(succeeded(&result));
        assert!(!root.path().join("hook-ran").exists());
        assert!(
            command(
                root.path(),
                &["show", "--pretty=format:", "--name-only", "HEAD"]
            )?
            .contains("a.txt")
        );
        assert_eq!(
            command(root.path(), &["diff", "--cached", "--name-only"])?.trim(),
            "b.txt"
        );
        let checkout = GitCheckout
            .execute(
                &context(root.path())?,
                json!({"branch":"new-branch","create":true}),
            )
            .await;
        assert!(matches!(checkout, Err(ToolError::Conflict(_))));
        Ok(())
    }
}
