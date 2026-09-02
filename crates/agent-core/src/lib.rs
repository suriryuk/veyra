use agent_context::{
    ContextInput, ContextManager, ContextProfile, ContextReport, MemorySnippet,
    RepositoryRetriever, RetrievalReport, WorkspaceRetriever,
};
use agent_model::{
    Message, ModelEventSink, ModelProvider, ModelRequest, RequestedToolCall, SamplingConfig,
    TokenUsage,
};
use agent_security::{
    ApprovalDecision, ApprovalProvider, ApprovalRequest, AuditEvent, AuditPhase, AuditSink,
    RiskLevel, SessionId, TaskId, ToolCallId, WorkspaceGuard, approval_fingerprint,
};
use agent_tools::{ExecutionLimits, ToolContext, ToolError, ToolRegistry, ToolResult};
use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowPhase {
    Discovery,
    Editing,
    Verifying,
    Recovering,
    Reviewing,
    Completed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    Compiler,
    Test,
    Timeout,
    PatchConflict,
    PolicyViolation,
    Command,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureInfo {
    pub kind: FailureKind,
    pub fingerprint: String,
    pub occurrences: usize,
    pub replan_required: bool,
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
    pub workflow_phase: WorkflowPhase,
    pub failure: Option<FailureInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub workspace: String,
    pub task: String,
    pub status: AgentStatus,
    pub plan: Vec<PlanStep>,
    pub messages: Vec<Message>,
    pub observations: Vec<Observation>,
    pub iteration: usize,
    pub tool_calls: usize,
    pub consecutive_errors: usize,
    pub workflow_phase: WorkflowPhase,
    pub change_sequence: usize,
    pub last_successful_verification: Option<usize>,
    pub last_diff_review: Option<usize>,
    pub failure_counts: BTreeMap<String, usize>,
}

#[async_trait]
pub trait SessionRepository: Send + Sync {
    async fn checkpoint(
        &self,
        state: &AgentState,
        event: Option<&AgentEvent>,
    ) -> Result<(), String>;

    async fn relevant_memories(
        &self,
        workspace: &str,
        task: &str,
        limit: usize,
    ) -> Result<Vec<MemorySnippet>, String>;

    async fn store_memory(&self, state: &AgentState, answer: &str) -> Result<(), String>;
}

#[derive(Debug, Clone)]
pub struct AgentLimits {
    pub max_iterations: usize,
    pub max_tool_calls: usize,
    pub max_consecutive_errors: usize,
    pub max_identical_failures: usize,
}

impl Default for AgentLimits {
    fn default() -> Self {
        Self {
            max_iterations: 30,
            max_tool_calls: 50,
            max_consecutive_errors: 3,
            max_identical_failures: 3,
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
    WorkflowPhaseChanged {
        task_id: TaskId,
        phase: WorkflowPhase,
    },
    FailureClassified {
        task_id: TaskId,
        failure: FailureInfo,
    },
    ContextBuilt {
        task_id: TaskId,
        report: ContextReport,
        retrieval: RetrievalReport,
    },
    ContextUsageObserved {
        task_id: TaskId,
        profile: agent_context::ContextProfileName,
        estimated_prompt_tokens: usize,
        usage: Option<TokenUsage>,
        estimation_delta: Option<i64>,
        overflow_retry: bool,
    },
    ToolRequested {
        call_id: ToolCallId,
        model_call_id: String,
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
    #[error("context failed: {0}")]
    Context(String),
    #[error("model context overflowed after one compact retry: {0}")]
    ContextOverflow(String),
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("tool failed: {0}")]
    Tool(String),
    #[error("audit failed: {0}")]
    Audit(String),
    #[error("session persistence failed: {0}")]
    Persistence(String),
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
    context: ContextManager,
    system_prompt: String,
    cancellation: CancellationToken,
    sessions: Option<Arc<dyn SessionRepository>>,
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
    pub context_profile: ContextProfile,
    pub system_prompt: String,
    pub cancellation: CancellationToken,
    pub sessions: Option<Arc<dyn SessionRepository>>,
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
            context: ContextManager::new(config.context_profile),
            system_prompt: config.system_prompt,
            cancellation: config.cancellation,
            sessions: config.sessions,
        }
    }

    pub async fn run(&self, task: impl Into<String>) -> Result<AgentState, AgentError> {
        self.run_in_session(SessionId::new(), Vec::new(), task)
            .await
    }

    pub async fn run_in_session(
        &self,
        session_id: SessionId,
        mut history: Vec<Message>,
        task: impl Into<String>,
    ) -> Result<AgentState, AgentError> {
        let task = task.into();
        let task_id = TaskId::new();
        if history.is_empty() {
            history.push(Message::system(self.system_prompt.clone()));
        }
        history.push(Message::user(task.clone()));
        let mut state = AgentState {
            session_id,
            task_id,
            workspace: self.workspace.root().display().to_string(),
            task: task.clone(),
            status: AgentStatus::Ready,
            plan: vec![PlanStep {
                id: Uuid::new_v4(),
                description: task.clone(),
                status: StepStatus::InProgress,
            }],
            messages: history,
            observations: Vec::new(),
            iteration: 0,
            tool_calls: 0,
            consecutive_errors: 0,
            workflow_phase: WorkflowPhase::Discovery,
            change_sequence: 0,
            last_successful_verification: None,
            last_diff_review: None,
            failure_counts: BTreeMap::new(),
        };
        self.emit(
            &state,
            AgentEvent::PlanCreated {
                task_id,
                steps: state.plan.clone(),
            },
        )
        .await?;
        self.set_status(&mut state, AgentStatus::Thinking).await;
        self.run_state(state).await
    }

    pub async fn resume(&self, mut state: AgentState) -> Result<AgentState, AgentError> {
        if state.workspace != self.workspace.root().display().to_string() {
            return Err(AgentError::Persistence(
                "stored session workspace does not match the configured workspace".to_owned(),
            ));
        }
        if matches!(
            state.status,
            AgentStatus::Completed | AgentStatus::Failed | AgentStatus::Cancelled
        ) {
            return Err(AgentError::Persistence(
                "terminal tasks cannot be resumed; start a new task in the session".to_owned(),
            ));
        }
        self.set_status(&mut state, AgentStatus::Thinking).await;
        self.run_state(state).await
    }

    async fn run_state(&self, mut state: AgentState) -> Result<AgentState, AgentError> {
        let task_id = state.task_id;

        let retrieval = WorkspaceRetriever::new(self.workspace.root())
            .retrieve(
                &state.task,
                match self.context.profile().name {
                    agent_context::ContextProfileName::Large => 128,
                    _ => 64,
                },
            )
            .map_err(|error| AgentError::Context(error.to_string()))?;
        let sources = retrieval.snippets;
        let retrieval_report = retrieval.report;
        let memories = if let Some(repository) = &self.sessions {
            repository
                .relevant_memories(&state.workspace, &state.task, 8)
                .await
                .map_err(AgentError::Persistence)?
        } else {
            Vec::new()
        };

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
            let tools = self.registry.definitions();
            let plan = state
                .plan
                .iter()
                .map(|step| format!("- [{:?}] {}", step.status, step.description))
                .collect::<Vec<_>>();
            let mut overflow_retry = false;
            let (response, context_report) = loop {
                let built = self
                    .context
                    .build(ContextInput {
                        system_prompt: &self.system_prompt,
                        task: &state.task,
                        plan: &plan,
                        history: &state.messages,
                        sources: &sources,
                        memories: &memories,
                        tools: &tools,
                        aggressive: overflow_retry,
                    })
                    .map_err(|error| AgentError::Context(error.to_string()))?;
                self.emit(
                    &state,
                    AgentEvent::ContextBuilt {
                        task_id,
                        report: built.report.clone(),
                        retrieval: retrieval_report.clone(),
                    },
                )
                .await?;
                let report = built.report;
                let response = self
                    .provider
                    .complete(
                        ModelRequest {
                            messages: built.messages,
                            tools: tools.clone(),
                            sampling: self.sampling.clone(),
                            max_output_tokens: Some(built.max_output_tokens),
                        },
                        &token_sink,
                    )
                    .await;
                if response
                    .as_ref()
                    .is_err_and(agent_model::ModelError::is_context_overflow)
                {
                    if overflow_retry {
                        let error = response.err().map_or_else(
                            || "unknown context overflow".to_owned(),
                            |error| error.to_string(),
                        );
                        return self
                            .fail(&mut state, AgentError::ContextOverflow(error))
                            .await;
                    }
                    overflow_retry = true;
                    continue;
                }
                break (response, report);
            };
            let response = match response {
                Ok(value) => {
                    state.consecutive_errors = 0;
                    let actual_prompt_tokens =
                        value.usage.as_ref().and_then(|usage| usage.prompt_tokens);
                    let estimation_delta = actual_prompt_tokens.map(|actual| {
                        i64::try_from(actual).unwrap_or(i64::MAX)
                            - i64::try_from(context_report.usage.prompt_tokens).unwrap_or(i64::MAX)
                    });
                    self.emit(
                        &state,
                        AgentEvent::ContextUsageObserved {
                            task_id,
                            profile: context_report.profile,
                            estimated_prompt_tokens: context_report.usage.prompt_tokens,
                            usage: value.usage.clone(),
                            estimation_delta,
                            overflow_retry,
                        },
                    )
                    .await?;
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
            self.checkpoint(&state, None).await?;
            if response.tool_calls.is_empty() {
                let answer = response
                    .content
                    .unwrap_or_else(|| "Task completed without a textual response.".to_owned());
                if let Some(requirement) = completion_requirement(&state) {
                    state.messages.push(Message::system(format!(
                        "Workflow evaluator rejected completion: {requirement}. Continue with the required verification or review before answering."
                    )));
                    continue;
                }
                if let Some(step) = state.plan.first_mut() {
                    step.status = StepStatus::Completed;
                }
                self.set_workflow_phase(&mut state, WorkflowPhase::Completed)
                    .await;
                self.set_status(&mut state, AgentStatus::Completed).await;
                self.emit(
                    &state,
                    AgentEvent::TaskCompleted {
                        task_id,
                        answer: answer.clone(),
                    },
                )
                .await?;
                if let Some(repository) = &self.sessions {
                    repository
                        .store_memory(&state, &answer)
                        .await
                        .map_err(AgentError::Persistence)?;
                }
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
                        let repeated_failure = value.failure.as_ref().is_some_and(|failure| {
                            failure.occurrences >= self.limits.max_identical_failures
                        });
                        state.consecutive_errors = if value.is_error {
                            state.consecutive_errors + 1
                        } else {
                            0
                        };
                        let content =
                            serde_json::to_string(&value).unwrap_or_else(|_| value.summary.clone());
                        state.messages.push(Message::tool(model_call_id, content));
                        state.observations.push(value);
                        self.checkpoint(&state, None).await?;
                        if repeated_failure {
                            return self
                                .fail(
                                    &mut state,
                                    AgentError::Limit(format!(
                                        "identical failure repeated {} times",
                                        self.limits.max_identical_failures
                                    )),
                                )
                                .await;
                        }
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
            self.emit(
                state,
                AgentEvent::ToolRequested {
                    call_id,
                    model_call_id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    risk: RiskLevel::Dangerous,
                },
            )
            .await?;
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
            self.emit(
                state,
                AgentEvent::ToolFailed {
                    call_id,
                    error: error.clone(),
                },
            )
            .await?;
            let failure = self
                .register_failure(state, FailureKind::Tool, &error)
                .await;
            return Ok(error_observation(
                call_id,
                error,
                state.workflow_phase,
                Some(failure),
            ));
        };
        let validation = tool.validate(&call.arguments);
        let risk_result = tool.risk(&call.arguments);
        let risk = risk_result
            .as_ref()
            .copied()
            .unwrap_or(RiskLevel::Dangerous);
        self.emit(
            state,
            AgentEvent::ToolRequested {
                call_id,
                model_call_id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                risk,
            },
        )
        .await?;
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
            let kind = failure_kind_for_tool_error(&error);
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
            self.emit(
                state,
                AgentEvent::ToolFailed {
                    call_id,
                    error: error.clone(),
                },
            )
            .await?;
            let failure = self.register_failure(state, kind, &error).await;
            return Ok(error_observation(
                call_id,
                error,
                state.workflow_phase,
                Some(failure),
            ));
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
            self.emit(
                state,
                AgentEvent::ApprovalRequested {
                    request: request.clone(),
                },
            )
            .await?;
            let decision = self.approver.decide(&request).await;
            self.emit(
                state,
                AgentEvent::ApprovalResolved {
                    approval_id: request.approval_id,
                    decision: decision.clone(),
                },
            )
            .await?;
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
                workflow_phase: state.workflow_phase,
                failure: None,
            });
        }
        let current_fingerprint =
            approval_fingerprint(&call.name, &call.arguments, self.workspace.root());
        if current_fingerprint != request.fingerprint {
            let error = "approval target changed".to_owned();
            let failure = self
                .register_failure(state, FailureKind::PolicyViolation, &error)
                .await;
            return Ok(error_observation(
                call_id,
                error,
                state.workflow_phase,
                Some(failure),
            ));
        }
        if let Err(error) = tool.validate(&call.arguments) {
            let kind = failure_kind_for_tool_error(&error);
            let error = error.to_string();
            let failure = self.register_failure(state, kind, &error).await;
            return Ok(error_observation(
                call_id,
                error,
                state.workflow_phase,
                Some(failure),
            ));
        }
        self.set_status(state, AgentStatus::ExecutingTool).await;
        self.emit(state, AgentEvent::ToolStarted { call_id })
            .await?;
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
                self.emit(
                    state,
                    AgentEvent::ToolCompleted {
                        call_id,
                        result: result.clone(),
                    },
                )
                .await?;
                Ok(self
                    .observation_from_result(state, call_id, &call.name, &call.arguments, result)
                    .await)
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
                self.emit(
                    state,
                    AgentEvent::ToolFailed {
                        call_id,
                        error: error.to_string(),
                    },
                )
                .await?;
                let kind = failure_kind_for_tool_error(&error);
                let error = error.to_string();
                let failure = self.register_failure(state, kind, &error).await;
                Ok(error_observation(
                    call_id,
                    error,
                    state.workflow_phase,
                    Some(failure),
                ))
            }
        }
    }

    async fn observation_from_result(
        &self,
        state: &mut AgentState,
        call_id: ToolCallId,
        name: &str,
        arguments: &Value,
        result: ToolResult,
    ) -> Observation {
        let success = result.metadata["success"].as_bool().unwrap_or(true);
        let failure = if success {
            None
        } else {
            let kind = failure_kind_from_metadata(&result.metadata);
            let seed = result.metadata["failure_fingerprint"]
                .as_str()
                .unwrap_or(&result.summary);
            Some(self.register_failure(state, kind, seed).await)
        };

        if success {
            match name {
                "patch_file" | "write_file" | "git_checkout" => {
                    state.change_sequence = state.change_sequence.saturating_add(1);
                    state.last_successful_verification = None;
                    state.last_diff_review = None;
                    self.set_workflow_phase(state, WorkflowPhase::Editing).await;
                }
                _ if is_verification_action(name, arguments) => {
                    state.last_successful_verification = Some(state.change_sequence);
                    state.last_diff_review = None;
                    self.set_workflow_phase(state, WorkflowPhase::Verifying)
                        .await;
                }
                "git_diff" => {
                    state.last_diff_review = Some(state.change_sequence);
                    self.set_workflow_phase(state, WorkflowPhase::Reviewing)
                        .await;
                }
                _ => {}
            }
        }

        Observation {
            tool_call_id: call_id,
            summary: result.summary,
            content: result.content,
            truncated: result.truncated,
            is_error: failure.is_some(),
            workflow_phase: state.workflow_phase,
            failure,
        }
    }

    async fn register_failure(
        &self,
        state: &mut AgentState,
        kind: FailureKind,
        seed: &str,
    ) -> FailureInfo {
        let fingerprint = if seed.len() == 64 && seed.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            seed.to_owned()
        } else {
            hex::encode(Sha256::digest(format!("{kind:?}:{seed}").as_bytes()))
        };
        let occurrences = state.failure_counts.entry(fingerprint.clone()).or_default();
        *occurrences = occurrences.saturating_add(1);
        let failure = FailureInfo {
            kind,
            fingerprint,
            occurrences: *occurrences,
            replan_required: *occurrences >= 2,
        };
        self.set_workflow_phase(state, WorkflowPhase::Recovering)
            .await;
        self.set_status(state, AgentStatus::Recovering).await;
        self.events
            .emit(AgentEvent::FailureClassified {
                task_id: state.task_id,
                failure: failure.clone(),
            })
            .await;
        failure
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

    async fn checkpoint(
        &self,
        state: &AgentState,
        event: Option<&AgentEvent>,
    ) -> Result<(), AgentError> {
        if let Some(repository) = &self.sessions {
            repository
                .checkpoint(state, event)
                .await
                .map_err(AgentError::Persistence)?;
        }
        Ok(())
    }

    async fn emit(&self, state: &AgentState, event: AgentEvent) -> Result<(), AgentError> {
        self.checkpoint(state, Some(&event)).await?;
        self.events.emit(event).await;
        Ok(())
    }

    async fn set_status(&self, state: &mut AgentState, status: AgentStatus) {
        state.status = status;
        let event = AgentEvent::StatusChanged {
            task_id: state.task_id,
            status,
        };
        if let Err(error) = self.emit(state, event).await {
            tracing::error!(error = %error, "failed to persist status transition");
        }
    }

    async fn set_workflow_phase(&self, state: &mut AgentState, phase: WorkflowPhase) {
        if state.workflow_phase == phase {
            return;
        }
        state.workflow_phase = phase;
        let event = AgentEvent::WorkflowPhaseChanged {
            task_id: state.task_id,
            phase,
        };
        if let Err(error) = self.emit(state, event).await {
            tracing::error!(error = %error, "failed to persist workflow transition");
        }
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
        self.emit(
            state,
            AgentEvent::TaskFailed {
                task_id: state.task_id,
                error: error.to_string(),
            },
        )
        .await?;
        Err(error)
    }
}

fn completion_requirement(state: &AgentState) -> Option<&'static str> {
    if state.change_sequence == 0 {
        return None;
    }
    if state.last_successful_verification != Some(state.change_sequence) {
        return Some("a successful post-change verification is missing");
    }
    if state.last_diff_review != Some(state.change_sequence) {
        return Some("a post-change git_diff review is missing");
    }
    None
}

fn is_verification_action(name: &str, arguments: &Value) -> bool {
    if matches!(name, "cargo_build" | "cargo_test") {
        return true;
    }
    if name != "run_command" {
        return false;
    }
    let Some(program) = arguments["program"].as_str() else {
        return false;
    };
    let executable = std::path::Path::new(program)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(program)
        .trim_end_matches(".exe")
        .to_ascii_lowercase();
    let args: Vec<String> = arguments["args"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_ascii_lowercase)
        .collect();
    match executable.as_str() {
        "cargo" => args.first().is_some_and(|argument| {
            matches!(
                argument.as_str(),
                "build" | "check" | "test" | "clippy" | "fmt"
            )
        }),
        "npm" | "pnpm" | "yarn" => args.iter().any(|argument| {
            matches!(
                argument.as_str(),
                "test" | "build" | "lint" | "typecheck" | "check"
            )
        }),
        "pytest" | "rustc" => true,
        "python" | "python3" => args
            .windows(2)
            .any(|pair| pair[0] == "-m" && matches!(pair[1].as_str(), "pytest" | "unittest")),
        "go" | "dotnet" => args
            .first()
            .is_some_and(|argument| matches!(argument.as_str(), "test" | "build")),
        "mvn" | "mvnw" | "gradle" | "gradlew" | "make" | "ninja" | "cmake" => args
            .iter()
            .any(|argument| matches!(argument.as_str(), "test" | "build" | "check" | "all")),
        _ => false,
    }
}

fn failure_kind_for_tool_error(error: &ToolError) -> FailureKind {
    match error {
        ToolError::Timeout(_) => FailureKind::Timeout,
        ToolError::Conflict(_) => FailureKind::PatchConflict,
        ToolError::Policy(_) | ToolError::InvalidArguments(_) => FailureKind::PolicyViolation,
        ToolError::Execution(_) => FailureKind::Command,
        ToolError::Io(_) | ToolError::Cancelled => FailureKind::Tool,
    }
}

fn failure_kind_from_metadata(metadata: &Value) -> FailureKind {
    match metadata["failure_kind"].as_str() {
        Some("compiler") => FailureKind::Compiler,
        Some("test") => FailureKind::Test,
        Some("timeout") => FailureKind::Timeout,
        Some("patch_conflict") => FailureKind::PatchConflict,
        Some("policy_violation") => FailureKind::PolicyViolation,
        _ => FailureKind::Command,
    }
}

fn error_observation(
    call_id: ToolCallId,
    error: String,
    workflow_phase: WorkflowPhase,
    failure: Option<FailureInfo>,
) -> Observation {
    Observation {
        tool_call_id: call_id,
        summary: error.clone(),
        content: json!({"error":error}),
        truncated: false,
        is_error: true,
        workflow_phase,
        failure,
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
    struct WorkflowTool {
        name: &'static str,
        risk: RiskLevel,
        results: Mutex<Vec<ToolResult>>,
    }
    #[async_trait]
    impl Tool for WorkflowTool {
        fn definition(&self) -> agent_model::ToolDefinition {
            agent_model::ToolDefinition::function(self.name, self.name, json!({"type":"object"}))
        }
        fn risk(&self, _: &Value) -> Result<RiskLevel, ToolError> {
            Ok(self.risk)
        }
        fn validate(&self, _: &Value) -> Result<(), ToolError> {
            Ok(())
        }
        async fn execute(&self, _: &ToolContext, _: Value) -> Result<ToolResult, ToolError> {
            self.results
                .lock()
                .map_err(|error| ToolError::Execution(error.to_string()))?
                .pop()
                .ok_or_else(|| ToolError::Execution("empty workflow result".to_owned()))
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

    struct OverflowModel {
        calls: AtomicUsize,
        always: bool,
    }

    #[async_trait]
    impl ModelProvider for OverflowModel {
        async fn complete(
            &self,
            request: ModelRequest,
            _: &dyn ModelEventSink,
        ) -> Result<ModelResponse, ModelError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(request.max_output_tokens, Some(2_048));
            if self.always || call == 0 {
                return Err(ModelError::Http {
                    status: 400,
                    body: "prompt exceeds the available context size".to_owned(),
                });
            }
            Ok(ModelResponse {
                content: Some("recovered".to_owned()),
                tool_calls: Vec::new(),
                usage: Some(TokenUsage {
                    prompt_tokens: Some(100),
                    completion_tokens: Some(5),
                    total_tokens: Some(105),
                }),
                finish_reason: FinishReason::Stop,
            })
        }

        async fn health(&self) -> Result<ModelHealth, ModelError> {
            Ok(ModelHealth {
                available: true,
                detail: "ok".to_owned(),
            })
        }
    }

    #[derive(Default)]
    struct ContextEvents {
        built: AtomicUsize,
        usage: AtomicUsize,
    }

    #[async_trait]
    impl AgentEventSink for ContextEvents {
        async fn emit(&self, event: AgentEvent) {
            match event {
                AgentEvent::ContextBuilt { .. } => {
                    self.built.fetch_add(1, Ordering::SeqCst);
                }
                AgentEvent::ContextUsageObserved { .. } => {
                    self.usage.fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
        }
    }

    fn context_runner(
        root: &std::path::Path,
        provider: Arc<dyn ModelProvider>,
        events: Arc<dyn AgentEventSink>,
    ) -> Result<AgentRunner, Box<dyn std::error::Error>> {
        Ok(AgentRunner::new(AgentRunnerConfig {
            provider,
            registry: Arc::new(ToolRegistry::new()),
            approver: Arc::new(Allow),
            audit: Arc::new(Null),
            events,
            workspace: WorkspaceGuard::new(root)?,
            limits: AgentLimits::default(),
            execution_limits: ExecutionLimits::default(),
            sampling: SamplingConfig::default(),
            context_profile: ContextProfile::default_32k(),
            system_prompt: "system".to_owned(),
            cancellation: CancellationToken::new(),
            sessions: None,
        }))
    }

    #[tokio::test]
    async fn context_overflow_is_compacted_and_retried_once()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let provider = Arc::new(OverflowModel {
            calls: AtomicUsize::new(0),
            always: false,
        });
        let events = Arc::new(ContextEvents::default());
        let runner = context_runner(temp.path(), provider.clone(), events.clone())?;
        let state = runner.run("answer briefly").await?;
        assert_eq!(state.status, AgentStatus::Completed);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert_eq!(events.built.load(Ordering::SeqCst), 2);
        assert_eq!(events.usage.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn second_context_overflow_fails_without_general_retry()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let provider = Arc::new(OverflowModel {
            calls: AtomicUsize::new(0),
            always: true,
        });
        let runner = context_runner(temp.path(), provider.clone(), Arc::new(Null))?;
        let error = match runner.run("answer briefly").await {
            Ok(_) => return Err("overflow unexpectedly succeeded".into()),
            Err(error) => error,
        };
        assert!(matches!(error, AgentError::ContextOverflow(_)));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        Ok(())
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
            context_profile: ContextProfile::default_32k(),
            system_prompt: "system".to_owned(),
            cancellation: CancellationToken::new(),
            sessions: None,
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
            context_profile: ContextProfile::default_32k(),
            system_prompt: "system".to_owned(),
            cancellation: CancellationToken::new(),
            sessions: None,
        });
        let state = runner.run("task").await?;
        assert!(!executed.load(Ordering::SeqCst));
        assert_eq!(state.observations.len(), 1);
        assert_eq!(state.observations[0].content["denied"], true);
        Ok(())
    }

    #[tokio::test]
    async fn workflow_requires_recovery_verification_and_diff_review()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let response = |name: &str| ModelResponse {
            content: None,
            tool_calls: vec![RequestedToolCall {
                id: format!("call-{name}"),
                name: name.to_owned(),
                arguments: json!({}),
            }],
            usage: None,
            finish_reason: FinishReason::ToolCalls,
        };
        let final_response = |content: &str| ModelResponse {
            content: Some(content.to_owned()),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Stop,
        };
        let provider = FakeModel {
            responses: Mutex::new(vec![
                final_response("complete"),
                response("git_diff"),
                final_response("premature"),
                response("cargo_build"),
                response("git_diff"),
                response("cargo_test"),
                response("patch_file"),
                response("cargo_test"),
                response("patch_file"),
            ]),
        };
        let success = |kind: &str| ToolResult {
            content: json!({"outcome":"completed"}),
            summary: format!("{kind} succeeded"),
            truncated: false,
            metadata: json!({"kind":kind,"success":true}),
        };
        let failure = ToolResult {
            content: json!({"outcome":"completed"}),
            summary: "cargo_test failed (test)".to_owned(),
            truncated: false,
            metadata: json!({"kind":"cargo_test","success":false,"failure_kind":"test","failure_fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}),
        };
        let mut registry = ToolRegistry::new();
        registry.register(WorkflowTool {
            name: "patch_file",
            risk: RiskLevel::Modify,
            results: Mutex::new(vec![success("patch_file"), success("patch_file")]),
        })?;
        registry.register(WorkflowTool {
            name: "cargo_test",
            risk: RiskLevel::Execute,
            results: Mutex::new(vec![success("cargo_test"), failure]),
        })?;
        registry.register(WorkflowTool {
            name: "cargo_build",
            risk: RiskLevel::Execute,
            results: Mutex::new(vec![success("cargo_build")]),
        })?;
        registry.register(WorkflowTool {
            name: "git_diff",
            risk: RiskLevel::Read,
            results: Mutex::new(vec![success("git_diff"), success("git_diff")]),
        })?;
        let runner = AgentRunner::new(AgentRunnerConfig {
            provider: Arc::new(provider),
            registry: Arc::new(registry),
            approver: Arc::new(Allow),
            audit: Arc::new(Null),
            events: Arc::new(Null),
            workspace: WorkspaceGuard::new(temp.path())?,
            limits: AgentLimits::default(),
            execution_limits: ExecutionLimits::default(),
            sampling: SamplingConfig::default(),
            context_profile: ContextProfile::default_32k(),
            system_prompt: "system".to_owned(),
            cancellation: CancellationToken::new(),
            sessions: None,
        });
        let state = runner.run("fix it").await?;
        assert_eq!(state.workflow_phase, WorkflowPhase::Completed);
        assert_eq!(state.change_sequence, 2);
        assert_eq!(state.last_successful_verification, Some(2));
        assert_eq!(state.last_diff_review, Some(2));
        assert!(state.observations.iter().any(|observation| {
            observation
                .failure
                .as_ref()
                .is_some_and(|failure| failure.kind == FailureKind::Test)
        }));
        Ok(())
    }

    #[tokio::test]
    async fn identical_failure_is_bounded() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let failed_call = || ModelResponse {
            content: None,
            tool_calls: vec![RequestedToolCall {
                id: Uuid::new_v4().to_string(),
                name: "cargo_test".to_owned(),
                arguments: json!({}),
            }],
            usage: None,
            finish_reason: FinishReason::ToolCalls,
        };
        let provider = FakeModel {
            responses: Mutex::new(vec![failed_call(), failed_call(), failed_call()]),
        };
        let failure = || ToolResult {
            content: json!({"outcome":"completed"}),
            summary: "same test failure".to_owned(),
            truncated: false,
            metadata: json!({"success":false,"failure_kind":"test","failure_fingerprint":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}),
        };
        let mut registry = ToolRegistry::new();
        registry.register(WorkflowTool {
            name: "cargo_test",
            risk: RiskLevel::Execute,
            results: Mutex::new(vec![failure(), failure(), failure()]),
        })?;
        let runner = AgentRunner::new(AgentRunnerConfig {
            provider: Arc::new(provider),
            registry: Arc::new(registry),
            approver: Arc::new(Allow),
            audit: Arc::new(Null),
            events: Arc::new(Null),
            workspace: WorkspaceGuard::new(temp.path())?,
            limits: AgentLimits::default(),
            execution_limits: ExecutionLimits::default(),
            sampling: SamplingConfig::default(),
            context_profile: ContextProfile::default_32k(),
            system_prompt: "system".to_owned(),
            cancellation: CancellationToken::new(),
            sessions: None,
        });
        let error = match runner.run("broken").await {
            Ok(_) => return Err("third identical failure did not stop".into()),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("identical failure repeated 3 times")
        );
        Ok(())
    }

    #[test]
    fn only_check_commands_count_as_generic_verification() {
        assert!(!is_verification_action(
            "run_command",
            &json!({"program":"cat","args":["src/main.rs"]})
        ));
        assert!(!is_verification_action(
            "run_command",
            &json!({"program":"find","args":["."]})
        ));
        assert!(is_verification_action(
            "run_command",
            &json!({"program":"npm","args":["run","typecheck"]})
        ));
        assert!(is_verification_action(
            "run_command",
            &json!({"program":"cargo","args":["clippy"]})
        ));
    }
}
