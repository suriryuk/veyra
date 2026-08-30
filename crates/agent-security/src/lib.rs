use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use uuid::Uuid;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name(pub Uuid);
        impl $name {
            #[must_use]
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

id_type!(SessionId);
id_type!(TaskId);
id_type!(ToolCallId);
id_type!(ApprovalId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RiskLevel {
    Read,
    Modify,
    Execute,
    Dangerous,
}

impl RiskLevel {
    #[must_use]
    pub fn requires_approval(self) -> bool {
        self != Self::Read
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub approval_id: ApprovalId,
    pub call_id: ToolCallId,
    pub action: String,
    pub risk: RiskLevel,
    pub operation: String,
    pub target: Option<String>,
    pub working_directory: Option<String>,
    pub reason: String,
    pub expected_effect: String,
    pub warning: Option<String>,
    pub fingerprint: String,
}

impl ApprovalRequest {
    #[must_use]
    pub fn for_tool(
        call_id: ToolCallId,
        tool_name: &str,
        risk: RiskLevel,
        arguments: &Value,
        workspace: &Path,
    ) -> Self {
        let fingerprint = approval_fingerprint(tool_name, arguments, workspace);
        let target = arguments
            .get("path")
            .or_else(|| arguments.get("program"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let warning = (risk == RiskLevel::Dangerous).then(|| {
            "DANGEROUS: this action may be destructive, privileged, or externally visible."
                .to_owned()
        });
        Self {
            approval_id: ApprovalId::new(),
            call_id,
            action: tool_name.to_owned(),
            risk,
            operation: format!("execute {tool_name}"),
            target,
            working_directory: Some(workspace.display().to_string()),
            reason: "The model requested this state-changing action.".to_owned(),
            expected_effect: summarize_effect(tool_name, arguments),
            warning,
            fingerprint,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum ApprovalDecision {
    NotRequired,
    AllowedOnce {
        decided_at: DateTime<Utc>,
        fingerprint: String,
    },
    Denied {
        decided_at: DateTime<Utc>,
    },
    Cancelled {
        decided_at: DateTime<Utc>,
    },
}

impl ApprovalDecision {
    #[must_use]
    pub fn permits(&self, expected_fingerprint: &str) -> bool {
        matches!(self, Self::AllowedOnce { fingerprint, .. } if fingerprint == expected_fingerprint)
    }
}

#[async_trait]
pub trait ApprovalProvider: Send + Sync {
    async fn decide(&self, request: &ApprovalRequest) -> ApprovalDecision;
}

#[derive(Debug, Error)]
pub enum SecurityError {
    #[error("workspace root is unavailable: {0}")]
    InvalidWorkspace(String),
    #[error("path is outside the workspace: {0}")]
    PathEscape(String),
    #[error("path contains a forbidden component: {0}")]
    InvalidPath(String),
    #[error("audit I/O failed: {0}")]
    AuditIo(#[from] std::io::Error),
    #[error("audit serialization failed: {0}")]
    AuditSerialization(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
pub struct WorkspaceGuard {
    root: PathBuf,
}

impl WorkspaceGuard {
    pub fn new(root: impl AsRef<Path>) -> Result<Self, SecurityError> {
        let canonical = std::fs::canonicalize(root.as_ref())
            .map_err(|error| SecurityError::InvalidWorkspace(error.to_string()))?;
        if !canonical.is_dir() {
            return Err(SecurityError::InvalidWorkspace(
                "root is not a directory".to_owned(),
            ));
        }
        Ok(Self { root: canonical })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn resolve_existing(&self, input: impl AsRef<Path>) -> Result<PathBuf, SecurityError> {
        let joined = self.join_checked(input.as_ref())?;
        let resolved = std::fs::canonicalize(&joined).map_err(|error| {
            SecurityError::InvalidPath(format!("{}: {error}", joined.display()))
        })?;
        self.ensure_inside(resolved)
    }

    pub fn resolve_new(&self, input: impl AsRef<Path>) -> Result<PathBuf, SecurityError> {
        let joined = self.join_checked(input.as_ref())?;
        if joined.exists() {
            return self.resolve_existing(joined);
        }
        let mut missing = Vec::new();
        let mut cursor = joined.as_path();
        while !cursor.exists() {
            let name = cursor
                .file_name()
                .ok_or_else(|| SecurityError::InvalidPath(joined.display().to_string()))?;
            missing.push(name.to_os_string());
            cursor = cursor
                .parent()
                .ok_or_else(|| SecurityError::InvalidPath(joined.display().to_string()))?;
        }
        let mut resolved = self.ensure_inside(
            std::fs::canonicalize(cursor)
                .map_err(|error| SecurityError::InvalidPath(error.to_string()))?,
        )?;
        for component in missing.iter().rev() {
            resolved.push(component);
        }
        self.ensure_inside(resolved)
    }

    fn join_checked(&self, input: &Path) -> Result<PathBuf, SecurityError> {
        if input.as_os_str().is_empty() {
            return Err(SecurityError::InvalidPath("empty path".to_owned()));
        }
        if input.is_absolute() {
            return self.ensure_inside(input.to_path_buf());
        }
        for component in input.components() {
            if matches!(
                component,
                Component::ParentDir | Component::Prefix(_) | Component::RootDir
            ) {
                return Err(SecurityError::InvalidPath(input.display().to_string()));
            }
        }
        Ok(self.root.join(input))
    }

    fn ensure_inside(&self, path: PathBuf) -> Result<PathBuf, SecurityError> {
        if path.starts_with(&self.root) {
            Ok(path)
        } else {
            Err(SecurityError::PathEscape(path.display().to_string()))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditPhase {
    Requested,
    ApprovalResolved,
    Started,
    Completed,
    Failed,
    Denied,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp: DateTime<Utc>,
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub call_id: ToolCallId,
    pub tool_name: String,
    pub arguments: Value,
    pub risk: RiskLevel,
    pub phase: AuditPhase,
    pub approval: Option<ApprovalDecision>,
    pub duration_ms: Option<u64>,
    pub summary: Option<String>,
    pub truncated: bool,
    pub error: Option<String>,
}

#[async_trait]
pub trait AuditSink: Send + Sync {
    async fn record(&self, event: AuditEvent) -> Result<(), SecurityError>;
}

#[derive(Clone)]
pub struct JsonlAuditSink {
    file: Arc<Mutex<File>>,
}

impl JsonlAuditSink {
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, SecurityError> {
        if let Some(parent) = path.as_ref().parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
        })
    }
}

#[async_trait]
impl AuditSink for JsonlAuditSink {
    async fn record(&self, mut event: AuditEvent) -> Result<(), SecurityError> {
        redact_value(&mut event.arguments);
        let mut line = serde_json::to_vec(&event)?;
        line.push(b'\n');
        let mut file = self.file.lock().await;
        file.write_all(&line).await?;
        file.flush().await?;
        Ok(())
    }
}

#[must_use]
pub fn approval_fingerprint(tool: &str, args: &Value, workspace: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(tool.as_bytes());
    hasher.update([0]);
    hasher.update(serde_json::to_vec(args).unwrap_or_default());
    hasher.update([0]);
    hasher.update(workspace.as_os_str().to_string_lossy().as_bytes());
    hex::encode(hasher.finalize())
}

pub fn redact_value(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, item) in map {
                if is_secret_key(key) {
                    *item = Value::String("[REDACTED]".to_owned());
                } else {
                    redact_value(item);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                redact_value(item);
            }
        }
        Value::String(text) if text.to_ascii_lowercase().starts_with("bearer ") => {
            *text = "[REDACTED]".to_owned();
        }
        _ => {}
    }
}

fn is_secret_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    [
        "authorization",
        "api_key",
        "apikey",
        "token",
        "password",
        "secret",
    ]
    .iter()
    .any(|needle| key.contains(needle))
}

fn summarize_effect(tool: &str, arguments: &Value) -> String {
    match tool {
        "patch_file" => format!(
            "patch {}",
            arguments
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("a file")
        ),
        "write_file" => format!(
            "write {}",
            arguments
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("a file")
        ),
        "run_command" => format!(
            "run {}",
            arguments
                .get("program")
                .and_then(Value::as_str)
                .unwrap_or("a program")
        ),
        _ => format!("execute {tool}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_parent_and_sibling_escape() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("work");
        std::fs::create_dir(&root)?;
        let guard = WorkspaceGuard::new(&root)?;
        assert!(guard.resolve_new("../outside").is_err());
        assert!(
            guard
                .resolve_new(temp.path().join("workspace-sibling"))
                .is_err()
        );
        assert!(guard.resolve_new("src/new.rs")?.starts_with(guard.root()));
        Ok(())
    }

    #[test]
    fn approval_is_bound_to_exact_arguments() {
        let one = serde_json::json!({"path":"a"});
        let two = serde_json::json!({"path":"b"});
        assert_ne!(
            approval_fingerprint("write_file", &one, Path::new("/w")),
            approval_fingerprint("write_file", &two, Path::new("/w"))
        );
    }

    #[test]
    fn redacts_nested_secrets() {
        let mut value = serde_json::json!({"environment":{"API_TOKEN":"secret"},"x":"Bearer abc"});
        redact_value(&mut value);
        assert_eq!(value["environment"]["API_TOKEN"], "[REDACTED]");
        assert_eq!(value["x"], "[REDACTED]");
    }

    #[tokio::test]
    async fn jsonl_audit_serializes_and_redacts() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("audit.jsonl");
        let sink = JsonlAuditSink::open(&path).await?;
        sink.record(AuditEvent {
            timestamp: Utc::now(),
            session_id: SessionId::new(),
            task_id: TaskId::new(),
            call_id: ToolCallId::new(),
            tool_name: "run_command".to_owned(),
            arguments: serde_json::json!({"api_key":"value"}),
            risk: RiskLevel::Execute,
            phase: AuditPhase::Requested,
            approval: None,
            duration_ms: None,
            summary: None,
            truncated: false,
            error: None,
        })
        .await?;
        let line = tokio::fs::read_to_string(path).await?;
        let value: Value = serde_json::from_str(line.trim())?;
        assert_eq!(value["arguments"]["api_key"], "[REDACTED]");
        assert_eq!(value["phase"], "requested");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn blocks_symlink_escape() -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("work");
        let outside = temp.path().join("outside");
        std::fs::create_dir(&root)?;
        std::fs::create_dir(&outside)?;
        std::fs::write(outside.join("secret"), "no")?;
        symlink(&outside, root.join("link"))?;
        let guard = WorkspaceGuard::new(&root)?;
        assert!(guard.resolve_existing("link/secret").is_err());
        assert!(guard.resolve_new("link/new").is_err());
        Ok(())
    }
}
