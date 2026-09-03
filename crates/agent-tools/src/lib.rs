mod builtins;
mod cargo_tools;
mod git_tools;
mod process;
mod web_tools;

use agent_model::ToolDefinition;
use agent_security::{RiskLevel, SessionId, TaskId, ToolCallId, WorkspaceGuard};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio_util::sync::CancellationToken;

pub use builtins::register_builtin_tools;
pub use web_tools::register_research_tools;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CommandProfile {
    Default,
    CargoBuild,
    CargoTest,
    Git,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandLimits {
    pub timeout_seconds: u64,
    pub stdout_limit_bytes: usize,
    pub stderr_limit_bytes: usize,
}

impl CommandLimits {
    #[must_use]
    pub fn bounded_timeout(&self, requested: Option<u64>) -> u64 {
        requested
            .unwrap_or(self.timeout_seconds)
            .min(self.timeout_seconds)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandProfiles {
    pub default: CommandLimits,
    pub cargo_build: CommandLimits,
    pub cargo_test: CommandLimits,
    pub git: CommandLimits,
}

impl Default for CommandProfiles {
    fn default() -> Self {
        Self {
            default: CommandLimits {
                timeout_seconds: 120,
                stdout_limit_bytes: 1_048_576,
                stderr_limit_bytes: 1_048_576,
            },
            cargo_build: CommandLimits {
                timeout_seconds: 300,
                stdout_limit_bytes: 2_097_152,
                stderr_limit_bytes: 2_097_152,
            },
            cargo_test: CommandLimits {
                timeout_seconds: 600,
                stdout_limit_bytes: 2_097_152,
                stderr_limit_bytes: 2_097_152,
            },
            git: CommandLimits {
                timeout_seconds: 60,
                stdout_limit_bytes: 1_048_576,
                stderr_limit_bytes: 1_048_576,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionLimits {
    pub command_timeout_seconds: u64,
    pub stdout_limit_bytes: usize,
    pub stderr_limit_bytes: usize,
    pub file_read_limit_bytes: usize,
    pub search_result_limit: usize,
    pub command_profiles: CommandProfiles,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            command_timeout_seconds: 120,
            stdout_limit_bytes: 1_048_576,
            stderr_limit_bytes: 1_048_576,
            file_read_limit_bytes: 2_097_152,
            search_result_limit: 500,
            command_profiles: CommandProfiles::default(),
        }
    }
}

impl ExecutionLimits {
    #[must_use]
    pub fn command_limits(&self, profile: CommandProfile) -> CommandLimits {
        let configured = match profile {
            CommandProfile::Default => &self.command_profiles.default,
            CommandProfile::CargoBuild => &self.command_profiles.cargo_build,
            CommandProfile::CargoTest => &self.command_profiles.cargo_test,
            CommandProfile::Git => &self.command_profiles.git,
        };
        let mut value = configured.clone();
        if profile == CommandProfile::Default {
            value.timeout_seconds = self.command_timeout_seconds;
            value.stdout_limit_bytes = self.stdout_limit_bytes;
            value.stderr_limit_bytes = self.stderr_limit_bytes;
        }
        value
    }
}

#[derive(Clone)]
pub struct ToolContext {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub call_id: ToolCallId,
    pub workspace: WorkspaceGuard,
    pub cancellation: CancellationToken,
    pub limits: ExecutionLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Value,
    pub summary: String,
    pub truncated: bool,
    pub metadata: Value,
}

impl ToolResult {
    #[must_use]
    pub fn text(text: String, summary: impl Into<String>, truncated: bool) -> Self {
        Self {
            content: Value::String(text),
            summary: summary.into(),
            truncated,
            metadata: Value::Object(Default::default()),
        }
    }
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),
    #[error("policy violation: {0}")]
    Policy(String),
    #[error("tool I/O failed: {0}")]
    Io(String),
    #[error("tool timed out after {0} seconds")]
    Timeout(u64),
    #[error("tool was cancelled")]
    Cancelled,
    #[error("file changed since approval: {0}")]
    Conflict(String),
    #[error("tool execution failed: {0}")]
    Execution(String),
}

impl From<agent_security::SecurityError> for ToolError {
    fn from(error: agent_security::SecurityError) -> Self {
        Self::Policy(error.to_string())
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn risk(&self, arguments: &Value) -> Result<RiskLevel, ToolError>;
    fn validate(&self, arguments: &Value) -> Result<(), ToolError>;
    async fn execute(
        &self,
        context: &ToolContext,
        arguments: Value,
    ) -> Result<ToolResult, ToolError>;
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T: Tool + 'static>(&mut self, tool: T) -> Result<(), ToolError> {
        self.register_arc(Arc::new(tool))
    }

    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) -> Result<(), ToolError> {
        let name = tool.definition().function.name;
        if self.tools.contains_key(&name) {
            return Err(ToolError::InvalidArguments(format!(
                "duplicate tool name: {name}"
            )));
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut definitions: Vec<_> = self.tools.values().map(|tool| tool.definition()).collect();
        definitions.sort_by(|a, b| a.function.name.cmp(&b.function.name));
        definitions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Stub;
    #[async_trait]
    impl Tool for Stub {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function("stub", "stub", json!({"type":"object"}))
        }
        fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
            Ok(RiskLevel::Read)
        }
        fn validate(&self, _: &Value) -> Result<(), ToolError> {
            Ok(())
        }
        async fn execute(&self, _: &ToolContext, _: Value) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::text(String::new(), "ok", false))
        }
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut registry = ToolRegistry::new();
        assert!(registry.register(Stub).is_ok());
        assert!(registry.register(Stub).is_err());
    }
}
