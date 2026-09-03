use agent_mcp::{McpConfig, McpManager, McpServerConfig};
use agent_security::{SessionId, TaskId, ToolCallId, WorkspaceGuard};
use agent_tools::{ExecutionLimits, ToolContext};
use serde_json::json;
use std::{collections::BTreeMap, io};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn stdio_handshake_paginated_discovery_and_call() -> Result<(), Box<dyn std::error::Error>> {
    let workspace = tempfile::tempdir()?;
    let config = McpConfig {
        servers: BTreeMap::from([(
            "fixture".to_owned(),
            McpServerConfig {
                enabled: true,
                command: env!("CARGO_BIN_EXE_agent-mcp-fake-server").to_owned(),
                ..McpServerConfig::default()
            },
        )]),
        ..McpConfig::default()
    };
    let guard = WorkspaceGuard::new(workspace.path())?;
    let connected = McpManager::connect_enabled(&config, &guard).await?;
    assert!(
        connected.diagnostics.is_empty(),
        "{:?}",
        connected.diagnostics
    );
    assert_eq!(connected.tools.len(), 2);
    let echo = connected
        .tools
        .iter()
        .find(|tool| tool.definition().function.name == "mcp__fixture__echo")
        .ok_or_else(|| io::Error::other("echo Tool should be discovered"))?;
    echo.validate(&json!({"text":"hello"}))?;
    let result = echo
        .execute(
            &ToolContext {
                session_id: SessionId::new(),
                task_id: TaskId::new(),
                call_id: ToolCallId::new(),
                workspace: guard,
                cancellation: CancellationToken::new(),
                limits: ExecutionLimits::default(),
            },
            json!({"text":"hello"}),
        )
        .await?;
    assert_eq!(result.content["text"], "fake stdio response");
    assert_eq!(result.metadata["server"], "fixture");
    connected.manager.shutdown().await;
    Ok(())
}
