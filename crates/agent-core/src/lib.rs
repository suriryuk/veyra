use agent_model::{
    Message, ModelEventSink, ModelProvider, ModelRequest, RequestedToolCall, SamplingConfig,
};
use agent_security::{
    ApprovalDecision, ApprovalProvider, ApprovalRequest, AuditEvent, AuditPhase, AuditSink,
    RiskLevel, SessionId, TaskId, ToolCallId, WorkspaceGuard, approval_fingerprint,
};
use agent_tools::{ExecutionLimits, ToolContext, ToolRegistry, ToolResult};
use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    Ready,
    Thinking,
    AwaitingApproval,
    ExecutingTool,
    Recovering,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: Uuid,
    pub description: String,
    pub status: StepStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub tool_call_id: ToolCallId,
    pub summary: String,
    pub content: Value,
    pub truncated: bool,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub task: String,
    pub status: AgentStatus,
    pub plan: Vec<PlanStep>,
    pub messages: Vec<Message>,
    pub observations: Vec<Observation>,
    pub iteration: usize,
    pub tool_calls: usize,
    pub consecutive_errors: usize,
}

#[derive(Debug, Clone)]
pub struct AgentLimits {
    pub max_iterations: usize,
    pub max_tool_calls: usize,
    pub max_consecutive_errors: usize,
}

impl Default for AgentLimits {
    fn default() -> Self {
        Self {
            max_iterations: 30,
            max_tool_calls: 50,
            max_consecutive_errors: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    StatusChanged {
        task_id: TaskId,
        status: AgentStatus,
    },
    TokenDelta {
        task_id: TaskId,
        text: String,
    },
    PlanCreated {
        task_id: TaskId,
        steps: Vec<PlanStep>,
    },
    ToolRequested {
        call_id: ToolCallId,
        name: String,
        arguments: Value,
        risk: RiskLevel,
    },
    ApprovalRequested {
        request: ApprovalRequest,
    },
    ApprovalResolved {
        approval_id: agent_security::ApprovalId,
        decision: ApprovalDecision,
    },
    ToolStarted {
        call_id: ToolCallId,
    },
    ToolCompleted {
        call_id: ToolCallId,
        result: ToolResult,
    },
    ToolFailed {
        call_id: ToolCallId,
        error: String,
    },
    TaskCompleted {
        task_id: TaskId,
        answer: String,
    },
    TaskFailed {
        task_id: TaskId,
        error: String,
    },
}

#[async_trait]
pub trait AgentEventSink: Send + Sync {
    async fn emit(&self, event: AgentEvent);
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("model failed: {0}")]
    Model(String),
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("tool failed: {0}")]
    Tool(String),
    #[error("audit failed: {0}")]
    Audit(String),
    #[error("agent limit reached: {0}")]
    Limit(String),
    #[error("task was cancelled")]
    Cancelled,
}

pub struct AgentRunner {
    provider: Arc<dyn ModelProvider>,
    registry: Arc<ToolRegistry>,
    approver: Arc<dyn ApprovalProvider>,
    audit: Arc<dyn AuditSink>,
    events: Arc<dyn AgentEventSink>,
    workspace: WorkspaceGuard,
    limits: AgentLimits,
    execution_limits: ExecutionLimits,
    sampling: SamplingConfig,
    system_prompt: String,
    cancellation: CancellationToken,
}

pub struct AgentRunnerConfig {
    pub provider: Arc<dyn ModelProvider>,
    pub registry: Arc<ToolRegistry>,
    pub approver: Arc<dyn ApprovalProvider>,
    pub audit: Arc<dyn AuditSink>,
    pub events: Arc<dyn AgentEventSink>,
    pub workspace: WorkspaceGuard,
    pub limits: AgentLimits,
    pub execution_limits: ExecutionLimits,
    pub sampling: SamplingConfig,
    pub system_prompt: String,
    pub cancellation: CancellationToken,
}

impl AgentRunner {
    #[must_use]
    pub fn new(config: AgentRunnerConfig) -> Self {
        Self {
            provider: config.provider,
            registry: config.registry,
            approver: config.approver,
            audit: config.audit,
            events: config.events,
            workspace: config.workspace,
            limits: config.limits,
            execution_limits: config.execution_limits,
            sampling: config.sampling,
            system_prompt: config.system_prompt,
            cancellation: config.cancellation,
        }
    }

    pub async fn run(&self, task: impl Into<String>) -> Result<AgentState, AgentError> {
        let task = task.into();
        let task_id = TaskId::new();
        let mut state = AgentState {
            session_id: SessionId::new(),
            task_id,
            task: task.clone(),
            status: AgentStatus::Ready,
            plan: vec![PlanStep {
                id: Uuid::new_v4(),
                description: task.clone(),
                status: StepStatus::InProgress,
            }],
            messages: vec![
                Message::system(self.system_prompt.clone()),
                Message::user(task),
            ],
            observations: Vec::new(),
            iteration: 0,
            tool_calls: 0,
            consecutive_errors: 0,
        };
        self.events
            .emit(AgentEvent::PlanCreated {
                task_id,
                steps: state.plan.clone(),
            })
            .await;
        self.set_status(&mut state, AgentStatus::Thinking).await;

        loop {
            if self.cancellation.is_cancelled() {
                return self.fail(&mut state, AgentError::Cancelled).await;
            }
            if state.iteration >= self.limits.max_iterations {
                return self
                    .fail(
                        &mut state,
                        AgentError::Limit(format!(
                            "max iterations ({})",
                            self.limits.max_iterations
                        )),
                    )
                    .await;
            }
            state.iteration += 1;
            let token_sink = ForwardTokens {
                task_id,
                sink: self.events.as_ref(),
            };
            let response = self
                .provider
                .complete(
                    ModelRequest {
                        messages: state.messages.clone(),
                        tools: self.registry.definitions(),
                        sampling: self.sampling.clone(),
                        max_output_tokens: None,
                    },
                    &token_sink,
                )
                .await;
            let response = match response {
                Ok(value) => {
                    state.consecutive_errors = 0;
                    value
                }
                Err(error) => {
                    state.consecutive_errors += 1;
                    if state.consecutive_errors >= self.limits.max_consecutive_errors {
                        return self
                            .fail(&mut state, AgentError::Model(error.to_string()))
                            .await;
                    }
                    self.set_status(&mut state, AgentStatus::Recovering).await;
                    state.messages.push(Message::system(format!(
                        "The previous model request failed transiently: {error}. Retry safely."
                    )));
                    self.set_status(&mut state, AgentStatus::Thinking).await;
                    continue;
                }
            };
            state.messages.push(Message::assistant_response(
                response.content.clone(),
                &response.tool_calls,
            ));
            if response.tool_calls.is_empty() {
                let answer = response
                    .content
                    .unwrap_or_else(|| "Task completed without a textual response.".to_owned());
                if let Some(step) = state.plan.first_mut() {
                    step.status = StepStatus::Completed;
                }
                self.set_status(&mut state, AgentStatus::Completed).await;
                self.events
                    .emit(AgentEvent::TaskCompleted { task_id, answer })
                    .await;
                return Ok(state);
            }
            for call in response.tool_calls {
                if state.tool_calls >= self.limits.max_tool_calls {
                    return self
                        .fail(
                            &mut state,
                            AgentError::Limit(format!(
                                "max tool calls ({})",
                                self.limits.max_tool_calls
                            )),
                        )
                        .await;
                }
                state.tool_calls += 1;
                let model_call_id = call.id.clone();
                let observation = self.execute_tool(&mut state, call).await;
                match observation {
                    Ok(value) => {
                        state.consecutive_errors = if value.is_error {
                            state.consecutive_errors + 1
                        } else {
                            0
                        };
                        let content =
                            serde_json::to_string(&value).unwrap_or_else(|_| value.summary.clone());
                        state.messages.push(Message::tool(model_call_id, content));
                        state.observations.push(value);
                    }
                    Err(error) => return self.fail(&mut state, error).await,
                }
                if state.consecutive_errors >= self.limits.max_consecutive_errors {
                    return self
                        .fail(
                            &mut state,
                            AgentError::Limit("max consecutive tool errors".to_owned()),
                        )
                        .await;
                }
            }
            self.set_status(&mut state, AgentStatus::Thinking).await;
        }
    }

    async fn execute_tool(
        &self,
        state: &mut AgentState,
        call: RequestedToolCall,
    ) -> Result<Observation, AgentError> {
        let call_id = ToolCallId::new();
        let Some(tool) = self.registry.get(&call.name) else {
            let error = format!("unknown tool: {}", call.name);
            self.events
                .emit(AgentEvent::ToolRequested {
                    call_id,
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    risk: RiskLevel::Dangerous,
                })
                .await;
            self.audit(
                state,
                call_id,
                &call.name,
                &call.arguments,
                RiskLevel::Dangerous,
                AuditPhase::Requested,
                None,
                None,
                None,
                false,
            )
            .await?;
            self.audit(
                state,
                call_id,
                &call.name,
                &call.arguments,
                RiskLevel::Dangerous,
                AuditPhase::Failed,
                None,
                None,
                Some(error.clone()),
                false,
            )
            .await?;
            self.events
                .emit(AgentEvent::ToolFailed {
                    call_id,
                    error: error.clone(),
                })
                .await;
            return Ok(error_observation(call_id, error));
        };
        let validation = tool.validate(&call.arguments);
        let risk_result = tool.risk(&call.arguments);
        let risk = risk_result
            .as_ref()
            .copied()
            .unwrap_or(RiskLevel::Dangerous);
        self.events
            .emit(AgentEvent::ToolRequested {
                call_id,
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                risk,
            })
            .await;
        self.audit(
            state,
            call_id,
            &call.name,
            &call.arguments,
            risk,
            AuditPhase::Requested,
            None,
            None,
            None,
            false,
        )
        .await?;
        if let Err(error) = validation.or(risk_result.map(|_| ())) {
            let error = error.to_string();
            self.audit(
                state,
                call_id,
                &call.name,
                &call.arguments,
                risk,
                AuditPhase::Failed,
                None,
                None,
                Some(error.clone()),
                false,
            )
            .await?;
            self.events
                .emit(AgentEvent::ToolFailed {
                    call_id,
                    error: error.clone(),
                })
                .await;
            return Ok(error_observation(call_id, error));
        }
        let request = ApprovalRequest::for_tool(
            call_id,
            &call.name,
            risk,
            &call.arguments,
            self.workspace.root(),
        );
        let decision = if risk.requires_approval() {
            self.set_status(state, AgentStatus::AwaitingApproval).await;
            self.events
                .emit(AgentEvent::ApprovalRequested {
                    request: request.clone(),
                })
                .await;
            let decision = self.approver.decide(&request).await;
            self.events
                .emit(AgentEvent::ApprovalResolved {
                    approval_id: request.approval_id,
                    decision: decision.clone(),
                })
                .await;
            decision
        } else {
            ApprovalDecision::NotRequired
        };
        self.audit(
            state,
            call_id,
            &call.name,
            &call.arguments,
            risk,
            AuditPhase::ApprovalResolved,
            Some(decision.clone()),
            None,
            None,
            false,
        )
        .await?;
        if risk.requires_approval() && !decision.permits(&request.fingerprint) {
            self.audit(
                state,
                call_id,
                &call.name,
                &call.arguments,
                risk,
                AuditPhase::Denied,
                Some(decision),
                None,
                Some("user denied execution".to_owned()),
                false,
            )
            .await?;
            return Ok(Observation {
                tool_call_id: call_id,
                summary: "tool execution denied by user".to_owned(),
                content: json!({"denied":true}),
                truncated: false,
                is_error: false,
            });
        }
        let current_fingerprint =
            approval_fingerprint(&call.name, &call.arguments, self.workspace.root());
        if current_fingerprint != request.fingerprint {
            return Ok(error_observation(
                call_id,
                "approval target changed".to_owned(),
            ));
        }
        if let Err(error) = tool.validate(&call.arguments) {
            return Ok(error_observation(call_id, error.to_string()));
        }
        self.set_status(state, AgentStatus::ExecutingTool).await;
        self.events.emit(AgentEvent::ToolStarted { call_id }).await;
        self.audit(
            state,
            call_id,
            &call.name,
            &call.arguments,
            risk,
            AuditPhase::Started,
            Some(decision.clone()),
            None,
            None,
            false,
        )
        .await?;
        let started = Instant::now();
        let context = ToolContext {
            session_id: state.session_id,
            task_id: state.task_id,
            call_id,
            workspace: self.workspace.clone(),
            cancellation: self.cancellation.clone(),
            limits: self.execution_limits.clone(),
        };
        match tool.execute(&context, call.arguments.clone()).await {
            Ok(result) => {
                let duration = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                self.audit(
                    state,
                    call_id,
                    &call.name,
                    &call.arguments,
                    risk,
                    AuditPhase::Completed,
                    Some(decision),
                    Some(duration),
                    Some(result.summary.clone()),
                    result.truncated,
                )
                .await?;
                self.events
                    .emit(AgentEvent::ToolCompleted {
                        call_id,
                        result: result.clone(),
                    })
                    .await;
                Ok(Observation {
                    tool_call_id: call_id,
                    summary: result.summary,
                    content: result.content,
                    truncated: result.truncated,
                    is_error: false,
                })
            }
            Err(error) => {
                let duration = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                self.audit(
                    state,
                    call_id,
                    &call.name,
                    &call.arguments,
                    risk,
                    AuditPhase::Failed,
                    Some(decision),
                    Some(duration),
                    Some(error.to_string()),
                    false,
                )
                .await?;
                self.events
                    .emit(AgentEvent::ToolFailed {
                        call_id,
                        error: error.to_string(),
                    })
                    .await;
                Ok(error_observation(call_id, error.to_string()))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn audit(
        &self,
        state: &AgentState,
        call_id: ToolCallId,
        name: &str,
        arguments: &Value,
        risk: RiskLevel,
        phase: AuditPhase,
        approval: Option<ApprovalDecision>,
        duration_ms: Option<u64>,
        summary: Option<String>,
        truncated: bool,
    ) -> Result<(), AgentError> {
        let error = matches!(phase, AuditPhase::Failed | AuditPhase::Denied)
            .then(|| summary.clone())
            .flatten();
        self.audit
            .record(AuditEvent {
                timestamp: Utc::now(),
                session_id: state.session_id,
                task_id: state.task_id,
                call_id,
                tool_name: name.to_owned(),
                arguments: arguments.clone(),
                risk,
                phase,
                approval,
                duration_ms,
                summary,
                truncated,
                error,
            })
            .await
            .map_err(|e| AgentError::Audit(e.to_string()))
    }

    async fn set_status(&self, state: &mut AgentState, status: AgentStatus) {
        state.status = status;
        self.events
            .emit(AgentEvent::StatusChanged {
                task_id: state.task_id,
                status,
            })
            .await;
    }

    async fn fail(
        &self,
        state: &mut AgentState,
        error: AgentError,
    ) -> Result<AgentState, AgentError> {
        if let Some(step) = state.plan.first_mut() {
            step.status = StepStatus::Failed;
        }
        self.set_status(
            state,
            if matches!(error, AgentError::Cancelled) {
                AgentStatus::Cancelled
            } else {
                AgentStatus::Failed
            },
        )
        .await;
        self.events
            .emit(AgentEvent::TaskFailed {
                task_id: state.task_id,
                error: error.to_string(),
            })
            .await;
        Err(error)
    }
}

fn error_observation(call_id: ToolCallId, error: String) -> Observation {
    Observation {
        tool_call_id: call_id,
        summary: error.clone(),
        content: json!({"error":error}),
        truncated: false,
        is_error: true,
    }
}

struct ForwardTokens<'a> {
    task_id: TaskId,
    sink: &'a dyn AgentEventSink,
}
#[async_trait]
impl ModelEventSink for ForwardTokens<'_> {
    async fn token(&self, text: &str) {
        self.sink
            .emit(AgentEvent::TokenDelta {
                task_id: self.task_id,
                text: text.to_owned(),
            })
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_model::{FinishReason, ModelError, ModelHealth, ModelResponse};
    use agent_security::SecurityError;
    use agent_tools::{Tool, ToolError};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct FakeModel {
        responses: Mutex<Vec<ModelResponse>>,
    }
    #[async_trait]
    impl ModelProvider for FakeModel {
        async fn complete(
            &self,
            _: ModelRequest,
            sink: &dyn ModelEventSink,
        ) -> Result<ModelResponse, ModelError> {
            sink.token("done").await;
            self.responses
                .lock()
                .map_err(|e| ModelError::Transport(e.to_string()))?
                .pop()
                .ok_or_else(|| ModelError::Transport("empty".to_owned()))
        }
        async fn health(&self) -> Result<ModelHealth, ModelError> {
            Ok(ModelHealth {
                available: true,
                detail: "ok".to_owned(),
            })
        }
    }
    struct Allow;
    #[async_trait]
    impl ApprovalProvider for Allow {
        async fn decide(&self, r: &ApprovalRequest) -> ApprovalDecision {
            ApprovalDecision::AllowedOnce {
                decided_at: Utc::now(),
                fingerprint: r.fingerprint.clone(),
            }
        }
    }
    struct Deny;
    #[async_trait]
    impl ApprovalProvider for Deny {
        async fn decide(&self, _: &ApprovalRequest) -> ApprovalDecision {
            ApprovalDecision::Denied {
                decided_at: Utc::now(),
            }
        }
    }
    struct SpyTool(Arc<AtomicBool>);
    #[async_trait]
    impl Tool for SpyTool {
        fn definition(&self) -> agent_model::ToolDefinition {
            agent_model::ToolDefinition::function("spy_write", "spy", json!({"type":"object"}))
        }
        fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
            Ok(RiskLevel::Modify)
        }
        fn validate(&self, _: &Value) -> Result<(), ToolError> {
            Ok(())
        }
        async fn execute(&self, _: &ToolContext, _: Value) -> Result<ToolResult, ToolError> {
            self.0.store(true, Ordering::SeqCst);
            Ok(ToolResult::text("changed".to_owned(), "changed", false))
        }
    }
    struct Null;
    #[async_trait]
    impl AgentEventSink for Null {
        async fn emit(&self, _: AgentEvent) {}
    }
    #[async_trait]
    impl AuditSink for Null {
        async fn record(&self, _: AuditEvent) -> Result<(), SecurityError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn final_response_completes_single_step() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let provider = FakeModel {
            responses: Mutex::new(vec![ModelResponse {
                content: Some("answer".to_owned()),
                tool_calls: vec![],
                usage: None,
                finish_reason: FinishReason::Stop,
            }]),
        };
        let runner = AgentRunner::new(AgentRunnerConfig {
            provider: Arc::new(provider),
            registry: Arc::new(ToolRegistry::new()),
            approver: Arc::new(Allow),
            audit: Arc::new(Null),
            events: Arc::new(Null),
            workspace: WorkspaceGuard::new(temp.path())?,
            limits: AgentLimits::default(),
            execution_limits: ExecutionLimits::default(),
            sampling: SamplingConfig::default(),
            system_prompt: "system".to_owned(),
            cancellation: CancellationToken::new(),
        });
        let state = runner.run("task").await?;
        assert_eq!(state.status, AgentStatus::Completed);
        assert_eq!(state.plan[0].status, StepStatus::Completed);
        Ok(())
    }

    #[tokio::test]
    async fn denied_modify_tool_is_never_executed() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let executed = Arc::new(AtomicBool::new(false));
        let mut registry = ToolRegistry::new();
        registry.register(SpyTool(executed.clone()))?;
        let provider = FakeModel {
            responses: Mutex::new(vec![
                ModelResponse {
                    content: Some("denial respected".to_owned()),
                    tool_calls: vec![],
                    usage: None,
                    finish_reason: FinishReason::Stop,
                },
                ModelResponse {
                    content: None,
                    tool_calls: vec![RequestedToolCall {
                        id: "model-call".to_owned(),
                        name: "spy_write".to_owned(),
                        arguments: json!({}),
                    }],
                    usage: None,
                    finish_reason: FinishReason::ToolCalls,
                },
            ]),
        };
        let runner = AgentRunner::new(AgentRunnerConfig {
            provider: Arc::new(provider),
            registry: Arc::new(registry),
            approver: Arc::new(Deny),
            audit: Arc::new(Null),
            events: Arc::new(Null),
            workspace: WorkspaceGuard::new(temp.path())?,
            limits: AgentLimits::default(),
            execution_limits: ExecutionLimits::default(),
            sampling: SamplingConfig::default(),
            system_prompt: "system".to_owned(),
            cancellation: CancellationToken::new(),
        });
        let state = runner.run("task").await?;
        assert!(!executed.load(Ordering::SeqCst));
        assert_eq!(state.observations.len(), 1);
        assert_eq!(state.observations[0].content["denied"], true);
        Ok(())
    }
}
