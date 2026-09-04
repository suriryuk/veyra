use agent_context::{ContextProfile, ContextProfileName};
use agent_core::{AgentEvent, AgentEventSink, AgentLimits, AgentRunner, AgentRunnerConfig};
use agent_document::{
    DocumentLimits, DocumentRepository, DocumentSearchHit, DocumentSearchQuery, DocumentStatus,
    DocumentSummary, IndexResult,
};
use agent_mcp::{McpConfig, McpManager};
use agent_model::{
    ModelFleetHealth, ModelId, ModelManager, ModelProfile, ModelProvider, ModelRoute, ModelRoutes,
    RouterModelManager, SamplingConfig, ToolDefinition,
};
use agent_research::{FetchPolicy, HttpFetcher, SearxngProvider};
use agent_security::{
    ApprovalDecision, ApprovalId, ApprovalProvider, ApprovalRequest, CompositeAuditSink,
    JsonlAuditSink, SessionId, TaskId, WorkspaceGuard,
};
use agent_storage::{ResolveApproval, SqliteSessionRepository, StoredEvent};
use agent_tools::{
    CommandLimits, CommandProfiles, ExecutionLimits, ToolRegistry, register_builtin_tools,
    register_document_tools, register_research_tools, register_vision_tool,
};
use agent_vision::{
    OpenAiVisionProvider, PopplerRenderer, VisionLimits, VisionPdfFallback, VisionService,
};
use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{Mutex, broadcast, oneshot};
use tokio_util::sync::CancellationToken;

const SYSTEM_PROMPT: &str = include_str!("../../../prompts/system.md");

#[derive(Debug, Error)]
pub enum AppError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("storage failed: {0}")]
    Storage(String),
    #[error("security failed: {0}")]
    Security(String),
    #[error("runtime failed: {0}")]
    Runtime(String),
    #[error("session not found: {0}")]
    SessionNotFound(String),
    #[error("session already has an active task")]
    SessionBusy,
    #[error("approval not found")]
    ApprovalNotFound,
    #[error("approval was already resolved")]
    ApprovalConflict,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextProfileChoice {
    Default,
    Large,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerSection {
    pub bind: String,
    pub allow_remote: bool,
    pub frontend_directory: PathBuf,
}

impl Default for ServerSection {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:3000".to_owned(),
            allow_remote: false,
            frontend_directory: PathBuf::from("frontend/dist"),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct AppConfig {
    pub agent: AgentSection,
    pub model: ModelSection,
    pub context: Option<ContextSection>,
    pub documents: DocumentLimits,
    pub vision: VisionLimits,
    pub research: ResearchSection,
    pub mcp: McpConfig,
    pub security: SecuritySection,
    pub tools: ToolsSection,
    pub logging: LoggingSection,
    pub storage: StorageSection,
    pub server: ServerSection,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ContextSection {
    pub profile: ContextProfileChoice,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ResearchSection {
    pub searxng_base_url: String,
    pub request_timeout_seconds: u64,
    pub max_redirects: usize,
    pub max_response_bytes: usize,
    pub max_results: usize,
    pub user_agent: String,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AgentSection {
    pub max_iterations: usize,
    pub max_consecutive_errors: usize,
    pub max_tool_calls: usize,
    pub max_identical_failures: usize,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ModelSection {
    pub base_url: String,
    pub model: String,
    pub context_size: usize,
    pub request_timeout_seconds: u64,
    pub sampling: SamplingSection,
    pub routes: ModelRoutesSection,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ModelRoutesSection {
    pub default: Option<String>,
    pub large: Option<String>,
    pub vision: Option<String>,
    pub load_timeout_seconds: u64,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SamplingSection {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SecuritySection {
    pub workspace_root: PathBuf,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ToolsSection {
    pub command_timeout_seconds: u64,
    pub stdout_limit_bytes: usize,
    pub stderr_limit_bytes: usize,
    pub file_read_limit_bytes: usize,
    pub search_result_limit: usize,
    pub command_profiles: CommandProfilesSection,
}
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct CommandProfilesSection {
    pub default: CommandLimitOverrides,
    pub cargo_build: CommandLimitOverrides,
    pub cargo_test: CommandLimitOverrides,
    pub git: CommandLimitOverrides,
}
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct CommandLimitOverrides {
    pub timeout_seconds: Option<u64>,
    pub stdout_limit_bytes: Option<usize>,
    pub stderr_limit_bytes: Option<usize>,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LoggingSection {
    pub level: String,
    pub directory: PathBuf,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct StorageSection {
    pub database_path: PathBuf,
}

impl Default for ContextSection {
    fn default() -> Self {
        Self {
            profile: ContextProfileChoice::Default,
        }
    }
}
impl Default for ResearchSection {
    fn default() -> Self {
        Self {
            searxng_base_url: "http://127.0.0.1:8888/".into(),
            request_timeout_seconds: 20,
            max_redirects: 5,
            max_response_bytes: 2_097_152,
            max_results: 10,
            user_agent: "Veyra/0.9".into(),
        }
    }
}
impl Default for AgentSection {
    fn default() -> Self {
        Self {
            max_iterations: 30,
            max_consecutive_errors: 3,
            max_tool_calls: 50,
            max_identical_failures: 3,
        }
    }
}
impl Default for ModelSection {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8080/v1".into(),
            model: "Qwen3-Coder-30B-A3B-Instruct".into(),
            context_size: 32768,
            request_timeout_seconds: 300,
            sampling: SamplingSection::default(),
            routes: ModelRoutesSection::default(),
        }
    }
}
impl Default for ModelRoutesSection {
    fn default() -> Self {
        Self {
            default: None,
            large: None,
            vision: None,
            load_timeout_seconds: 300,
        }
    }
}
impl Default for SamplingSection {
    fn default() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.8,
            top_k: 20,
            repeat_penalty: 1.05,
        }
    }
}
impl Default for SecuritySection {
    fn default() -> Self {
        Self {
            workspace_root: PathBuf::from("workspace"),
        }
    }
}
impl Default for ToolsSection {
    fn default() -> Self {
        Self {
            command_timeout_seconds: 120,
            stdout_limit_bytes: 1_048_576,
            stderr_limit_bytes: 1_048_576,
            file_read_limit_bytes: 2_097_152,
            search_result_limit: 500,
            command_profiles: CommandProfilesSection::default(),
        }
    }
}
impl Default for LoggingSection {
    fn default() -> Self {
        Self {
            level: "info".into(),
            directory: PathBuf::from("logs"),
        }
    }
}
impl Default for StorageSection {
    fn default() -> Self {
        Self {
            database_path: PathBuf::from("data/veyra.sqlite3"),
        }
    }
}

impl AppConfig {
    pub fn load(
        path: Option<&Path>,
        workspace_override: Option<PathBuf>,
    ) -> Result<Self, AppError> {
        let selected = path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("config/agent.toml"));
        let mut value = if selected.exists() {
            let text =
                std::fs::read_to_string(&selected).map_err(|e| AppError::Config(e.to_string()))?;
            toml::from_str(&text)
                .map_err(|e| AppError::Config(format!("{}: {e}", selected.display())))?
        } else if path.is_some() {
            return Err(AppError::Config(format!(
                "file not found: {}",
                selected.display()
            )));
        } else {
            Self::default()
        };
        if let Ok(v) = env::var("VEYRA_MODEL_BASE_URL") {
            value.model.base_url = v;
        }
        if let Ok(v) = env::var("VEYRA_MODEL_NAME") {
            value.model.model = v;
        }
        if let Ok(v) = env::var("VEYRA_VISION_MODEL_NAME") {
            value.model.routes.vision = Some(v);
        }
        if let Ok(v) = env::var("VEYRA_WORKSPACE_ROOT") {
            value.security.workspace_root = PathBuf::from(v);
        }
        if let Ok(v) = env::var("VEYRA_LOG_LEVEL") {
            value.logging.level = v;
        }
        if let Ok(v) = env::var("VEYRA_SEARXNG_BASE_URL") {
            value.research.searxng_base_url = v;
        }
        if let Some(v) = workspace_override {
            value.security.workspace_root = v;
        }
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), AppError> {
        if self.agent.max_iterations == 0
            || self.agent.max_tool_calls == 0
            || self.agent.max_consecutive_errors == 0
            || self.agent.max_identical_failures == 0
        {
            return Err(AppError::Config("agent limits must be positive".into()));
        }
        if self.model.request_timeout_seconds == 0
            || self.model.routes.load_timeout_seconds == 0
            || self.tools.command_timeout_seconds == 0
        {
            return Err(AppError::Config("timeouts must be positive".into()));
        }
        if self.security.workspace_root.as_os_str().is_empty()
            || self.storage.database_path.as_os_str().is_empty()
        {
            return Err(AppError::Config(
                "workspace and database paths must not be empty".into(),
            ));
        }
        self.documents
            .validate()
            .map_err(|e| AppError::Config(e.to_string()))?;
        self.vision
            .validate()
            .map_err(|e| AppError::Config(e.to_string()))?;
        self.mcp
            .validate()
            .map_err(|e| AppError::Config(e.to_string()))?;
        Ok(())
    }

    #[must_use]
    pub fn context_profile(&self, choice: Option<ContextProfileChoice>) -> ContextProfile {
        match choice.or_else(|| self.context.as_ref().map(|c| c.profile)) {
            Some(ContextProfileChoice::Default) => ContextProfile::default_32k(),
            Some(ContextProfileChoice::Large) => ContextProfile::large_65k(),
            None if self.model.context_size == 32_768 => ContextProfile::default_32k(),
            None => ContextProfile::legacy(self.model.context_size),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StartedTask {
    pub session_id: String,
    pub task_id: String,
}

struct ActiveTask {
    task_id: String,
    cancellation: CancellationToken,
}

#[derive(Clone)]
pub struct ApplicationService {
    config: Arc<AppConfig>,
    database: Arc<SqliteSessionRepository>,
    approvals: Arc<ApprovalBroker>,
    events: broadcast::Sender<StoredEvent>,
    active: Arc<Mutex<HashMap<String, ActiveTask>>>,
}

impl ApplicationService {
    pub async fn open(config: AppConfig) -> Result<Self, AppError> {
        tokio::fs::create_dir_all(&config.security.workspace_root)
            .await
            .map_err(|e| AppError::Config(e.to_string()))?;
        let database = Arc::new(
            SqliteSessionRepository::open(&config.storage.database_path)
                .await
                .map_err(AppError::Storage)?,
        );
        let (events, _) = broadcast::channel(1024);
        Ok(Self {
            config: Arc::new(config),
            approvals: Arc::new(ApprovalBroker::new(database.clone())),
            database,
            events,
            active: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    #[must_use]
    pub fn config(&self) -> &AppConfig {
        &self.config
    }
    #[must_use]
    pub fn database(&self) -> Arc<SqliteSessionRepository> {
        self.database.clone()
    }
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<StoredEvent> {
        self.events.subscribe()
    }

    pub fn resolve_workspace(&self, relative: Option<&str>) -> Result<WorkspaceGuard, AppError> {
        let allowed = WorkspaceGuard::new(&self.config.security.workspace_root)
            .map_err(|e| AppError::Security(e.to_string()))?;
        let path = match relative
            .map(str::trim)
            .filter(|v| !v.is_empty() && *v != ".")
        {
            Some(value) => allowed
                .resolve_existing(value)
                .map_err(|e| AppError::Security(e.to_string()))?,
            None => allowed.root().to_path_buf(),
        };
        WorkspaceGuard::new(path).map_err(|e| AppError::Security(e.to_string()))
    }

    pub async fn create_session(&self, workspace: Option<&str>) -> Result<String, AppError> {
        let guard = self.resolve_workspace(workspace)?;
        let id = SessionId::new().to_string();
        self.database
            .create_session(&id, &guard.root().display().to_string())
            .await
            .map_err(AppError::Storage)?;
        Ok(id)
    }

    pub async fn start_task(
        &self,
        session: SessionId,
        message: String,
        profile: Option<ContextProfileChoice>,
    ) -> Result<StartedTask, AppError> {
        if message.trim().is_empty() {
            return Err(AppError::Config("message must not be empty".into()));
        }
        let key = session.to_string();
        let shown = self
            .database
            .show_session(&key, Some(1))
            .await
            .map_err(|_| AppError::SessionNotFound(key.clone()))?;
        let workspace = shown["session"]["workspace"]
            .as_str()
            .ok_or_else(|| AppError::SessionNotFound(key.clone()))?;
        let allowed = WorkspaceGuard::new(&self.config.security.workspace_root)
            .map_err(|e| AppError::Security(e.to_string()))?;
        let resolved = allowed
            .resolve_existing(workspace)
            .map_err(|e| AppError::Security(e.to_string()))?;
        let workspace =
            WorkspaceGuard::new(resolved).map_err(|e| AppError::Security(e.to_string()))?;
        let cancellation = CancellationToken::new();
        let task_id = TaskId::new();
        let history = self
            .database
            .load_latest(&key)
            .await
            .map(|s| s.messages)
            .unwrap_or_default();
        {
            let mut active = self.active.lock().await;
            if active.contains_key(&key) {
                return Err(AppError::SessionBusy);
            }
            active.insert(
                key.clone(),
                ActiveTask {
                    task_id: task_id.to_string(),
                    cancellation: cancellation.clone(),
                },
            );
        }
        let bundle = match build_runner(
            self.config.clone(),
            self.database.clone(),
            workspace,
            session,
            self.approvals.clone(),
            self.events.clone(),
            cancellation.clone(),
            self.config.context_profile(profile),
        )
        .await
        {
            Ok(bundle) => bundle,
            Err(error) => {
                self.active.lock().await.remove(&key);
                return Err(error);
            }
        };
        let service = self.clone();
        let task_key = key.clone();
        tokio::spawn(async move {
            let result = bundle
                .runner
                .run_in_session_with_task(session, task_id, history, message)
                .await;
            if let Err(error) = result {
                tracing::error!(session_id = %session, task_id = %task_id, error = %error, "agent task failed");
            }
            bundle.mcp.shutdown().await;
            service.active.lock().await.remove(&task_key);
        });
        Ok(StartedTask {
            session_id: key,
            task_id: task_id.to_string(),
        })
    }

    pub async fn cancel_task(&self, task_id: &str) -> Result<(), AppError> {
        let active = self.active.lock().await;
        let task = active
            .values()
            .find(|task| task.task_id == task_id)
            .ok_or_else(|| AppError::Runtime(format!("active task not found: {task_id}")))?;
        task.cancellation.cancel();
        Ok(())
    }

    pub async fn resolve_approval(&self, id: ApprovalId, allow: bool) -> Result<(), AppError> {
        self.approvals.resolve(id, allow).await
    }

    pub async fn model_status(&self) -> Result<ModelFleetHealth, AppError> {
        model_manager(&self.config)
            .map_err(|e| AppError::Runtime(e.to_string()))?
            .health()
            .await
            .map_err(|e| AppError::Runtime(e.to_string()))
    }

    pub async fn tools(&self) -> Result<Vec<ToolDefinition>, AppError> {
        let workspace = self.resolve_workspace(None)?;
        let manager = model_manager(&self.config).map_err(|e| AppError::Runtime(e.to_string()))?;
        let (registry, mcp) =
            registry(&self.config, &workspace, self.database.clone(), manager).await?;
        let definitions = registry.definitions();
        mcp.shutdown().await;
        Ok(definitions)
    }

    pub async fn documents(
        &self,
        workspace: &str,
        status: Option<DocumentStatus>,
        limit: usize,
    ) -> Result<Vec<DocumentSummary>, AppError> {
        self.database
            .list(workspace, status, limit)
            .await
            .map_err(|e| AppError::Storage(e.to_string()))
    }

    pub async fn search_documents(
        &self,
        query: DocumentSearchQuery,
    ) -> Result<Vec<DocumentSearchHit>, AppError> {
        self.database
            .search(query)
            .await
            .map_err(|e| AppError::Storage(e.to_string()))
    }

    pub async fn index_document(
        &self,
        workspace: &str,
        relative: &str,
    ) -> Result<IndexResult, AppError> {
        let guard =
            WorkspaceGuard::new(workspace).map_err(|e| AppError::Security(e.to_string()))?;
        let path = guard
            .resolve_existing(relative)
            .map_err(|e| AppError::Security(e.to_string()))?;
        let bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| AppError::Storage(e.to_string()))?;
        let service = agent_document::DocumentService::new(self.config.documents.clone())
            .map_err(|e| AppError::Config(e.to_string()))?;
        let manager = model_manager(&self.config).map_err(|e| AppError::Runtime(e.to_string()))?;
        let vision = vision_services(&self.config, manager)?;
        let document = service
            .parse_with_fallback(
                workspace,
                relative,
                &path,
                &bytes,
                vision.as_ref().map(|value| value.fallback.as_ref()),
                &CancellationToken::new(),
            )
            .await
            .map_err(|e| AppError::Storage(e.to_string()))?;
        self.database
            .upsert(&document)
            .await
            .map_err(|e| AppError::Storage(e.to_string()))
    }
}

struct ApprovalBroker {
    database: Arc<SqliteSessionRepository>,
    pending: Mutex<HashMap<String, oneshot::Sender<ApprovalDecision>>>,
}

impl ApprovalBroker {
    fn new(database: Arc<SqliteSessionRepository>) -> Self {
        Self {
            database,
            pending: Mutex::new(HashMap::new()),
        }
    }
    async fn resolve(&self, id: ApprovalId, allow: bool) -> Result<(), AppError> {
        let key = id.to_string();
        let decision = if allow {
            ApprovalDecision::AllowedOnce {
                decided_at: Utc::now(),
                fingerprint: self.approval_fingerprint(&key).await?,
            }
        } else {
            ApprovalDecision::Denied {
                decided_at: Utc::now(),
            }
        };
        match self
            .database
            .resolve_approval(&key, &decision)
            .await
            .map_err(AppError::Storage)?
        {
            ResolveApproval::NotFound => return Err(AppError::ApprovalNotFound),
            ResolveApproval::AlreadyResolved => return Err(AppError::ApprovalConflict),
            ResolveApproval::Resolved => {}
        }
        if let Some(sender) = self.pending.lock().await.remove(&key) {
            let _ = sender.send(decision);
        }
        Ok(())
    }
    async fn approval_fingerprint(&self, id: &str) -> Result<String, AppError> {
        let value = self
            .database
            .approval(id)
            .await
            .map_err(AppError::Storage)?
            .ok_or(AppError::ApprovalNotFound)?;
        value["fingerprint"]
            .as_str()
            .map(ToOwned::to_owned)
            .ok_or_else(|| AppError::Storage("approval fingerprint is missing".into()))
    }
}

struct BrokerApprover {
    broker: Arc<ApprovalBroker>,
    cancellation: CancellationToken,
}
#[async_trait]
impl ApprovalProvider for BrokerApprover {
    async fn decide(&self, request: &ApprovalRequest) -> ApprovalDecision {
        let (sender, receiver) = oneshot::channel();
        let key = request.approval_id.to_string();
        self.broker.pending.lock().await.insert(key.clone(), sender);
        if let Ok(Some(stored)) = self.broker.database.approval(&key).await
            && stored["status"] == "resolved"
            && let Ok(decision) =
                serde_json::from_value::<ApprovalDecision>(stored["decision"].clone())
        {
            self.broker.pending.lock().await.remove(&key);
            return decision;
        }
        tokio::select! {
            value = receiver => value.unwrap_or_else(|_| ApprovalDecision::Cancelled { decided_at: Utc::now() }),
            () = self.cancellation.cancelled() => ApprovalDecision::Cancelled { decided_at: Utc::now() },
        }
    }
}

struct ServerEvents {
    session_id: String,
    database: Arc<SqliteSessionRepository>,
    sender: broadcast::Sender<StoredEvent>,
}
#[async_trait]
impl AgentEventSink for ServerEvents {
    async fn emit(&self, _event: AgentEvent) {
        match self.database.latest_event(&self.session_id).await {
            Ok(Some(event)) => {
                let _ = self.sender.send(event);
            }
            Ok(None) => {}
            Err(error) => tracing::error!(%error, "failed to publish stored event"),
        }
    }
}

struct RunnerBundle {
    runner: AgentRunner,
    mcp: McpManager,
}

#[allow(clippy::too_many_arguments)]
async fn build_runner(
    config: Arc<AppConfig>,
    database: Arc<SqliteSessionRepository>,
    workspace: WorkspaceGuard,
    session: SessionId,
    approvals: Arc<ApprovalBroker>,
    events: broadcast::Sender<StoredEvent>,
    cancellation: CancellationToken,
    context: ContextProfile,
) -> Result<RunnerBundle, AppError> {
    let manager = model_manager(&config).map_err(|e| AppError::Runtime(e.to_string()))?;
    let (profile, route) = if context.name == ContextProfileName::Large {
        (ModelProfile::Large, ModelRoute::Large)
    } else {
        (ModelProfile::Default, ModelRoute::Default)
    };
    manager
        .switch_profile(profile)
        .await
        .map_err(|e| AppError::Runtime(e.to_string()))?;
    let provider = Arc::new(
        manager
            .provider(
                route,
                Duration::from_secs(config.model.request_timeout_seconds),
            )
            .map_err(|e| AppError::Runtime(e.to_string()))?,
    );
    provider
        .health()
        .await
        .map_err(|e| AppError::Runtime(e.to_string()))?;
    let (registry, mcp) = registry(&config, &workspace, database.clone(), manager).await?;
    let jsonl = Arc::new(
        JsonlAuditSink::open(config.logging.directory.join("audit.jsonl"))
            .await
            .map_err(|e| AppError::Runtime(e.to_string()))?,
    );
    let audit: Vec<Arc<dyn agent_security::AuditSink>> = vec![database.clone(), jsonl];
    let profiles = command_profiles(&config);
    let default_command = profiles.default.clone();
    let runner = AgentRunner::new(AgentRunnerConfig {
        provider,
        registry: Arc::new(registry),
        approver: Arc::new(BrokerApprover {
            broker: approvals,
            cancellation: cancellation.clone(),
        }),
        audit: Arc::new(CompositeAuditSink::new(audit)),
        events: Arc::new(ServerEvents {
            session_id: session.to_string(),
            database: database.clone(),
            sender: events,
        }),
        workspace,
        limits: AgentLimits {
            max_iterations: config.agent.max_iterations,
            max_tool_calls: config.agent.max_tool_calls,
            max_consecutive_errors: config.agent.max_consecutive_errors,
            max_identical_failures: config.agent.max_identical_failures,
        },
        execution_limits: ExecutionLimits {
            command_timeout_seconds: default_command.timeout_seconds,
            stdout_limit_bytes: default_command.stdout_limit_bytes,
            stderr_limit_bytes: default_command.stderr_limit_bytes,
            file_read_limit_bytes: config.tools.file_read_limit_bytes,
            search_result_limit: config.tools.search_result_limit,
            command_profiles: profiles,
        },
        sampling: SamplingConfig {
            temperature: config.model.sampling.temperature,
            top_p: config.model.sampling.top_p,
            top_k: config.model.sampling.top_k,
            repeat_penalty: config.model.sampling.repeat_penalty,
        },
        context_profile: context,
        system_prompt: SYSTEM_PROMPT.to_owned(),
        cancellation,
        sessions: Some(database),
    });
    Ok(RunnerBundle { runner, mcp })
}

async fn registry(
    config: &AppConfig,
    workspace: &WorkspaceGuard,
    database: Arc<SqliteSessionRepository>,
    manager: Arc<RouterModelManager>,
) -> Result<(ToolRegistry, McpManager), AppError> {
    let mut value = ToolRegistry::new();
    register_builtin_tools(&mut value).map_err(|e| AppError::Runtime(e.to_string()))?;
    let timeout = Duration::from_secs(config.research.request_timeout_seconds);
    let provider = Arc::new(
        SearxngProvider::new(
            &config.research.searxng_base_url,
            timeout,
            &config.research.user_agent,
        )
        .map_err(|e| AppError::Config(e.to_string()))?,
    );
    let fetcher = HttpFetcher::new(FetchPolicy::production(
        timeout,
        config.research.max_redirects,
        config.research.max_response_bytes,
        config.research.user_agent.clone(),
    ))
    .map_err(|e| AppError::Config(e.to_string()))?;
    register_research_tools(&mut value, provider, fetcher, config.research.max_results)
        .map_err(|e| AppError::Runtime(e.to_string()))?;
    let document_service = agent_document::DocumentService::new(config.documents.clone())
        .map_err(|e| AppError::Config(e.to_string()))?;
    let vision = vision_services(config, manager.clone())?;
    let fallback = vision.as_ref().map(|bundle| bundle.fallback.clone());
    register_document_tools(&mut value, database, document_service, fallback)
        .map_err(|e| AppError::Runtime(e.to_string()))?;
    if let Some(vision) = vision {
        register_vision_tool(&mut value, vision.service)
            .map_err(|e| AppError::Runtime(e.to_string()))?;
    }
    let connected = McpManager::connect_enabled(&config.mcp, workspace)
        .await
        .map_err(|e| AppError::Runtime(e.to_string()))?;
    for tool in connected.tools {
        value
            .register_arc(tool)
            .map_err(|e| AppError::Runtime(e.to_string()))?;
    }
    Ok((value, connected.manager))
}

fn model_manager(config: &AppConfig) -> Result<Arc<RouterModelManager>, agent_model::ModelError> {
    let default = config
        .model
        .routes
        .default
        .clone()
        .unwrap_or_else(|| config.model.model.clone());
    let large = config
        .model
        .routes
        .large
        .clone()
        .unwrap_or_else(|| default.clone());
    Ok(Arc::new(RouterModelManager::new(
        &config.model.base_url,
        ModelRoutes {
            default: ModelId(default),
            large: ModelId(large),
            vision: config.model.routes.vision.clone().map(ModelId),
        },
        Duration::from_secs(config.model.request_timeout_seconds),
        Duration::from_secs(config.model.routes.load_timeout_seconds),
    )?))
}

struct VisionServices {
    service: Arc<VisionService>,
    fallback: Arc<dyn agent_document::ScannedPdfFallback>,
}
fn vision_services(
    config: &AppConfig,
    manager: Arc<RouterModelManager>,
) -> Result<Option<VisionServices>, AppError> {
    let Some(model) = manager.model_for(ModelRoute::Vision) else {
        return Ok(None);
    };
    let provider = Arc::new(
        OpenAiVisionProvider::new(
            &config.model.base_url,
            manager,
            Duration::from_secs(config.model.request_timeout_seconds),
            config.vision.max_output_chars,
        )
        .map_err(|e| AppError::Runtime(e.to_string()))?,
    );
    let service = Arc::new(
        VisionService::new(config.vision.clone(), provider.clone())
            .map_err(|e| AppError::Runtime(e.to_string()))?,
    );
    let renderer = Arc::new(
        PopplerRenderer::new(config.vision.clone())
            .map_err(|e| AppError::Runtime(e.to_string()))?,
    );
    let fallback = Arc::new(VisionPdfFallback::new(
        renderer,
        provider,
        config.vision.clone(),
        model.as_str(),
    ));
    Ok(Some(VisionServices { service, fallback }))
}

fn command_profiles(config: &AppConfig) -> CommandProfiles {
    let defaults = CommandProfiles::default();
    let flat = CommandLimits {
        timeout_seconds: config.tools.command_timeout_seconds,
        stdout_limit_bytes: config.tools.stdout_limit_bytes,
        stderr_limit_bytes: config.tools.stderr_limit_bytes,
    };
    CommandProfiles {
        default: apply_overrides(flat, &config.tools.command_profiles.default),
        cargo_build: apply_overrides(
            defaults.cargo_build,
            &config.tools.command_profiles.cargo_build,
        ),
        cargo_test: apply_overrides(
            defaults.cargo_test,
            &config.tools.command_profiles.cargo_test,
        ),
        git: apply_overrides(defaults.git, &config.tools.command_profiles.git),
    }
}
fn apply_overrides(mut limits: CommandLimits, value: &CommandLimitOverrides) -> CommandLimits {
    if let Some(v) = value.timeout_seconds {
        limits.timeout_seconds = v;
    }
    if let Some(v) = value.stdout_limit_bytes {
        limits.stdout_limit_bytes = v;
    }
    if let Some(v) = value.stderr_limit_bytes {
        limits.stderr_limit_bytes = v;
    }
    limits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn session_workspace_is_confined_to_the_allowed_root()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let allowed = temp.path().join("allowed");
        let child = allowed.join("repository");
        std::fs::create_dir_all(&child)?;
        let mut config = AppConfig::default();
        config.security.workspace_root = allowed.clone();
        config.storage.database_path = temp.path().join("veyra.db");
        config.logging.directory = temp.path().join("logs");
        let service = ApplicationService::open(config).await?;
        let id = service.create_session(Some("repository")).await?;
        let shown = service.database().show_session(&id, Some(1)).await?;
        assert_eq!(
            shown["session"]["workspace"],
            std::fs::canonicalize(child)?.display().to_string()
        );
        assert!(service.create_session(Some("../outside")).await.is_err());
        Ok(())
    }
}
