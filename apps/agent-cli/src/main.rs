use agent_context::ContextProfile;
use agent_core::{
    AgentError, AgentEvent, AgentEventSink, AgentLimits, AgentRunner, AgentRunnerConfig, AgentState,
};
use agent_document::{
    DocumentLimits, DocumentRepository, DocumentSearchQuery, DocumentService, DocumentStatus,
    collect_supported_paths,
};
use agent_mcp::{McpConfig, McpManager};
use agent_model::Message;
use agent_model::{ModelProvider, OpenAiCompatibleProvider, SamplingConfig};
use agent_research::{FetchPolicy, HttpFetcher, SearxngProvider};
use agent_security::{
    ApprovalDecision, ApprovalProvider, ApprovalRequest, CompositeAuditSink, JsonlAuditSink,
    SessionId, WorkspaceGuard,
};
use agent_storage::SqliteSessionRepository;
use agent_tools::{
    CommandLimits, CommandProfiles, ExecutionLimits, ToolRegistry, register_builtin_tools,
    register_document_tools, register_research_tools,
};
use async_trait::async_trait;
use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use serde::Deserialize;
use std::env;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::prelude::*;

const SYSTEM_PROMPT: &str = include_str!("../../../prompts/system.md");

#[derive(Parser)]
#[command(name = "veyra", version, about = "A safe local coding agent")]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    workspace: Option<PathBuf>,
    #[arg(long, global = true, value_enum)]
    context_profile: Option<ContextProfileArg>,
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Chat,
    Run {
        task: String,
    },
    Models {
        #[command(subcommand)]
        command: ModelCommand,
    },
    Tools {
        #[command(subcommand)]
        command: ToolCommand,
    },
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    Sessions {
        #[command(subcommand)]
        command: SessionCommand,
    },
    Documents {
        #[command(subcommand)]
        command: DocumentCommand,
    },
}

#[derive(Subcommand)]
enum ModelCommand {
    Status,
}
#[derive(Subcommand)]
enum ToolCommand {
    List,
}
#[derive(Subcommand)]
enum ConfigCommand {
    Check,
}
#[derive(Subcommand)]
enum SessionCommand {
    List {
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    Show {
        id: String,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        all: bool,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        research: bool,
    },
    Resume {
        id: String,
    },
    Prune {
        #[arg(long)]
        older_than: i64,
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Subcommand)]
enum DocumentCommand {
    Add {
        paths: Vec<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    List {
        #[arg(long, value_enum)]
        status: Option<DocumentStatusArg>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    Show {
        id: String,
        #[arg(long)]
        chunks: bool,
        #[arg(long)]
        json: bool,
    },
    Search {
        query: String,
        #[arg(long = "document")]
        documents: Vec<String>,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DocumentStatusArg {
    Ready,
    Partial,
    UnsupportedFormat,
    UnsupportedScanned,
    UnsupportedEncrypted,
    Failed,
}
impl From<DocumentStatusArg> for DocumentStatus {
    fn from(value: DocumentStatusArg) -> Self {
        match value {
            DocumentStatusArg::Ready => Self::Ready,
            DocumentStatusArg::Partial => Self::Partial,
            DocumentStatusArg::UnsupportedFormat => Self::UnsupportedFormat,
            DocumentStatusArg::UnsupportedScanned => Self::UnsupportedScanned,
            DocumentStatusArg::UnsupportedEncrypted => Self::UnsupportedEncrypted,
            DocumentStatusArg::Failed => Self::Failed,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ContextProfileArg {
    Default,
    Large,
}

#[derive(Debug, Error)]
enum CliError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("tool setup failed: {0}")]
    Tool(#[from] agent_tools::ToolError),
    #[error("security setup failed: {0}")]
    Security(#[from] agent_security::SecurityError),
    #[error("model setup failed: {0}")]
    Model(#[from] agent_model::ModelError),
    #[error("agent failed: {0}")]
    Agent(#[from] agent_core::AgentError),
    #[error("storage failed: {0}")]
    Storage(String),
    #[error("MCP setup failed: {0}")]
    Mcp(#[from] agent_mcp::McpError),
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct AppConfig {
    agent: AgentSection,
    model: ModelSection,
    context: Option<ContextSection>,
    documents: DocumentLimits,
    research: ResearchSection,
    mcp: McpConfig,
    security: SecuritySection,
    tools: ToolsSection,
    logging: LoggingSection,
    storage: StorageSection,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct ContextSection {
    profile: ContextProfileArg,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct ResearchSection {
    searxng_base_url: String,
    request_timeout_seconds: u64,
    max_redirects: usize,
    max_response_bytes: usize,
    max_results: usize,
    user_agent: String,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct AgentSection {
    max_iterations: usize,
    max_consecutive_errors: usize,
    max_tool_calls: usize,
    max_identical_failures: usize,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct ModelSection {
    base_url: String,
    model: String,
    context_size: usize,
    request_timeout_seconds: u64,
    sampling: SamplingSection,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct SamplingSection {
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct SecuritySection {
    workspace_root: PathBuf,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct ToolsSection {
    command_timeout_seconds: u64,
    stdout_limit_bytes: usize,
    stderr_limit_bytes: usize,
    file_read_limit_bytes: usize,
    search_result_limit: usize,
    command_profiles: CommandProfilesSection,
}
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct CommandProfilesSection {
    default: CommandLimitOverrides,
    cargo_build: CommandLimitOverrides,
    cargo_test: CommandLimitOverrides,
    git: CommandLimitOverrides,
}
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct CommandLimitOverrides {
    timeout_seconds: Option<u64>,
    stdout_limit_bytes: Option<usize>,
    stderr_limit_bytes: Option<usize>,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct LoggingSection {
    level: String,
    directory: PathBuf,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct StorageSection {
    database_path: PathBuf,
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
impl Default for ContextSection {
    fn default() -> Self {
        Self {
            profile: ContextProfileArg::Default,
        }
    }
}
impl Default for ResearchSection {
    fn default() -> Self {
        Self {
            searxng_base_url: "http://127.0.0.1:8888/".to_owned(),
            request_timeout_seconds: 20,
            max_redirects: 5,
            max_response_bytes: 2_097_152,
            max_results: 10,
            user_agent: "Veyra/0.7".to_owned(),
        }
    }
}
impl Default for ModelSection {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8080/v1".to_owned(),
            model: "Qwen3-Coder-30B-A3B-Instruct".to_owned(),
            context_size: 32768,
            request_timeout_seconds: 300,
            sampling: SamplingSection::default(),
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
            level: "info".to_owned(),
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
    fn load(path: Option<&Path>, workspace_override: Option<PathBuf>) -> Result<Self, CliError> {
        let selected = path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("config/agent.toml"));
        let mut config = if selected.exists() {
            let contents = std::fs::read_to_string(&selected)?;
            toml::from_str(&contents)
                .map_err(|e| CliError::Config(format!("{}: {e}", selected.display())))?
        } else if path.is_some() {
            return Err(CliError::Config(format!(
                "file not found: {}",
                selected.display()
            )));
        } else {
            Self::default()
        };
        if let Ok(value) = env::var("VEYRA_MODEL_BASE_URL") {
            config.model.base_url = value;
        }
        if let Ok(value) = env::var("VEYRA_MODEL_NAME") {
            config.model.model = value;
        }
        if let Ok(value) = env::var("VEYRA_WORKSPACE_ROOT") {
            config.security.workspace_root = PathBuf::from(value);
        }
        if let Ok(value) = env::var("VEYRA_LOG_LEVEL") {
            config.logging.level = value;
        }
        if let Ok(value) = env::var("VEYRA_SEARXNG_BASE_URL") {
            config.research.searxng_base_url = value;
        }
        if let Some(value) = workspace_override {
            config.security.workspace_root = value;
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), CliError> {
        if self.agent.max_iterations == 0
            || self.agent.max_tool_calls == 0
            || self.agent.max_consecutive_errors == 0
            || self.agent.max_identical_failures == 0
        {
            return Err(CliError::Config("agent limits must be positive".to_owned()));
        }
        if self.model.request_timeout_seconds == 0 || self.tools.command_timeout_seconds == 0 {
            return Err(CliError::Config("timeouts must be positive".to_owned()));
        }
        if self.research.request_timeout_seconds == 0
            || self.research.max_redirects == 0
            || self.research.max_response_bytes == 0
            || self.research.max_results == 0
            || self.research.user_agent.trim().is_empty()
        {
            return Err(CliError::Config(
                "research limits and user agent must be non-empty and positive".to_owned(),
            ));
        }
        self.documents
            .validate()
            .map_err(|error| CliError::Config(error.to_string()))?;
        SearxngProvider::new(
            &self.research.searxng_base_url,
            Duration::from_secs(self.research.request_timeout_seconds),
            &self.research.user_agent,
        )
        .map_err(|error| CliError::Config(error.to_string()))?;
        self.mcp.validate()?;
        for profile in [
            &self.tools.command_profiles.default,
            &self.tools.command_profiles.cargo_build,
            &self.tools.command_profiles.cargo_test,
            &self.tools.command_profiles.git,
        ] {
            if profile.timeout_seconds == Some(0)
                || profile.stdout_limit_bytes == Some(0)
                || profile.stderr_limit_bytes == Some(0)
            {
                return Err(CliError::Config(
                    "command profile limits must be positive".to_owned(),
                ));
            }
        }
        if self.model.context_size < 1024 {
            return Err(CliError::Config(
                "model.context_size is too small".to_owned(),
            ));
        }
        if self.security.workspace_root.as_os_str().is_empty() {
            return Err(CliError::Config("workspace root is empty".to_owned()));
        }
        if self.storage.database_path.as_os_str().is_empty() {
            return Err(CliError::Config(
                "storage database path is empty".to_owned(),
            ));
        }
        Ok(())
    }

    fn resolved_context_profile(
        &self,
        override_profile: Option<ContextProfileArg>,
    ) -> ContextProfile {
        match override_profile.or_else(|| self.context.as_ref().map(|context| context.profile)) {
            Some(ContextProfileArg::Default) => ContextProfile::default_32k(),
            Some(ContextProfileArg::Large) => ContextProfile::large_65k(),
            None if self.model.context_size == 32_768 => ContextProfile::default_32k(),
            None => ContextProfile::legacy(self.model.context_size),
        }
    }
}

struct CliApprover {
    cancellation: CancellationToken,
}

enum ApprovalInput {
    Line(String),
    Cancelled,
}

async fn wait_for_approval_input(
    cancellation: &CancellationToken,
    receiver: tokio::sync::oneshot::Receiver<io::Result<Option<String>>>,
) -> ApprovalInput {
    tokio::select! {
        () = cancellation.cancelled() => ApprovalInput::Cancelled,
        result = receiver => ApprovalInput::Line(
            result.ok().and_then(Result::ok).flatten().unwrap_or_default()
        ),
    }
}

#[async_trait]
impl ApprovalProvider for CliApprover {
    async fn decide(&self, request: &ApprovalRequest) -> ApprovalDecision {
        eprintln!(
            "\nPermission required: {:?} — {}",
            request.risk, request.expected_effect
        );
        eprintln!("operation: {}", request.operation);
        if let Some(target) = &request.target {
            eprintln!("target: {target}");
        }
        if let Some(cwd) = &request.working_directory {
            eprintln!("working directory: {cwd}");
        }
        if let Some(warning) = &request.warning {
            eprintln!("WARNING: {warning}");
        }
        eprint!("Allow once? [y/N] ");
        let _ = io::stderr().flush();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        std::thread::spawn(move || {
            let _ = sender.send(read_console_line());
        });
        let input = match wait_for_approval_input(&self.cancellation, receiver).await {
            ApprovalInput::Line(input) => input,
            ApprovalInput::Cancelled => {
                eprintln!("\napproval cancelled");
                return ApprovalDecision::Cancelled {
                    decided_at: Utc::now(),
                };
            }
        };
        let allowed = matches!(input.trim().to_ascii_lowercase().as_str(), "y" | "yes");
        if allowed {
            ApprovalDecision::AllowedOnce {
                decided_at: Utc::now(),
                fingerprint: request.fingerprint.clone(),
            }
        } else {
            ApprovalDecision::Denied {
                decided_at: Utc::now(),
            }
        }
    }
}

struct CliEvents {
    streamed_current_response: AtomicBool,
    stream_boundary_pending: AtomicBool,
}
impl CliEvents {
    fn new() -> Self {
        Self {
            streamed_current_response: AtomicBool::new(false),
            stream_boundary_pending: AtomicBool::new(false),
        }
    }

    fn record_token(&self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.streamed_current_response
            .store(true, Ordering::Relaxed);
        self.stream_boundary_pending.store(true, Ordering::Relaxed);
    }

    fn close_stream_line(&self) -> bool {
        if self.stream_boundary_pending.swap(false, Ordering::Relaxed) {
            println!();
            let _ = io::stdout().flush();
            true
        } else {
            false
        }
    }
}
#[async_trait]
impl AgentEventSink for CliEvents {
    async fn emit(&self, event: AgentEvent) {
        if !matches!(&event, AgentEvent::TokenDelta { .. }) {
            self.close_stream_line();
        }
        if matches!(&event, AgentEvent::ContextBuilt { .. }) {
            self.streamed_current_response
                .store(false, Ordering::Relaxed);
        }
        match event {
            AgentEvent::TokenDelta { text, .. } => {
                self.record_token(&text);
                print!("{text}");
                let _ = io::stdout().flush();
            }
            AgentEvent::ToolStarted { call_id } => eprintln!("\n[tool {call_id}] started"),
            AgentEvent::ToolCompleted { result, .. } => eprintln!(
                "[tool] {}{}",
                result.summary,
                tool_result_qualifier(&result)
            ),
            AgentEvent::ToolFailed { error, .. } => eprintln!("[tool] failed: {error}"),
            AgentEvent::WorkflowPhaseChanged { phase, .. } => eprintln!("[workflow] {phase:?}"),
            AgentEvent::FailureClassified { failure, .. } => eprintln!(
                "[workflow] {:?} failure occurrence {}{}",
                failure.kind,
                failure.occurrences,
                if failure.replan_required {
                    " — replan required"
                } else {
                    ""
                }
            ),
            AgentEvent::ContextBuilt {
                report, retrieval, ..
            } => eprintln!(
                "[context] profile={} prompt~{}/{} output={} workspace_sources={}/{} memories={}/{} messages={} compressed={} retrieval={}",
                report.profile.as_str(),
                report.usage.prompt_tokens,
                report.usage.input_limit,
                report.usage.output_reserve,
                report.selected_sources,
                report.selected_sources + report.dropped_sources,
                report.selected_memories,
                report.selected_memories + report.dropped_memories,
                report.selected_message_groups,
                report.compressed_observations,
                retrieval.backend,
            ),
            AgentEvent::ContextUsageObserved {
                estimated_prompt_tokens,
                usage,
                estimation_delta,
                overflow_retry,
                ..
            } => eprintln!(
                "[context] estimate={} actual={} delta={} retry={}",
                estimated_prompt_tokens,
                usage
                    .as_ref()
                    .and_then(|usage| usage.prompt_tokens)
                    .map_or_else(|| "unknown".to_owned(), |value| value.to_string()),
                estimation_delta.map_or_else(|| "unknown".to_owned(), |value| value.to_string()),
                overflow_retry,
            ),
            AgentEvent::TaskCompleted { answer, .. } => {
                if !self
                    .streamed_current_response
                    .swap(false, Ordering::Relaxed)
                {
                    println!("{answer}");
                }
            }
            AgentEvent::TaskFailed { error, .. } => {
                self.streamed_current_response
                    .store(false, Ordering::Relaxed);
                eprintln!("task failed: {error}");
            }
            _ => {}
        }
    }
}

fn tool_result_qualifier(result: &agent_tools::ToolResult) -> &'static str {
    if !result.truncated {
        ""
    } else if result.metadata["kind"] == "web_search" && result.metadata["limit_reached"] == true {
        " (configured limit reached)"
    } else {
        " (truncated)"
    }
}

#[tokio::main]
async fn main() -> Result<(), CliError> {
    let Cli {
        config: config_path,
        workspace,
        context_profile,
        command,
    } = Cli::parse();
    let config = AppConfig::load(config_path.as_deref(), workspace)?;
    let selected_context = config.resolved_context_profile(context_profile);
    let _log_guard = init_logging(&config)?;
    match command.unwrap_or(Commands::Chat) {
        Commands::Config {
            command: ConfigCommand::Check,
        } => {
            let database = storage(&config).await?;
            println!(
                "configuration is valid; context_profile={} context_limit={} output_reserve={} database={}",
                selected_context.name.as_str(),
                selected_context.budget.context_limit,
                selected_context.budget.output_reserve,
                database.path().display()
            );
            Ok(())
        }
        Commands::Tools {
            command: ToolCommand::List,
        } => {
            tokio::fs::create_dir_all(&config.security.workspace_root).await?;
            let workspace = WorkspaceGuard::new(&config.security.workspace_root)?;
            let database = Arc::new(storage(&config).await?);
            let (registry, manager) = registry(&config, &workspace, database).await?;
            for definition in registry.definitions() {
                println!(
                    "{}\t{}",
                    definition.function.name, definition.function.description
                );
            }
            manager.shutdown().await;
            Ok(())
        }
        Commands::Models {
            command: ModelCommand::Status,
        } => {
            let provider = provider(&config)?;
            let health = provider.health().await?;
            println!("available={} {}", health.available, health.detail);
            Ok(())
        }
        Commands::Run { task } => {
            let state = run_task(
                &config,
                &selected_context,
                SessionId::new(),
                Vec::new(),
                task,
            )
            .await?;
            println!("session_id={}", state.session_id);
            Ok(())
        }
        Commands::Chat => chat(&config, &selected_context).await,
        Commands::Sessions { command } => sessions(&config, &selected_context, command).await,
        Commands::Documents { command } => documents(&config, command).await,
    }
}

async fn chat(config: &AppConfig, context: &ContextProfile) -> Result<(), CliError> {
    chat_session(config, context, SessionId::new(), Vec::new()).await
}

async fn chat_session(
    config: &AppConfig,
    context: &ContextProfile,
    session_id: SessionId,
    mut history: Vec<Message>,
) -> Result<(), CliError> {
    println!(
        "Veyra v0.7 — session {}; context profile {}; enter a task, or /quit to exit.",
        session_id,
        context.name.as_str(),
    );
    loop {
        print!("> ");
        io::stdout().flush()?;
        let Some(task) = read_console_line()? else {
            break;
        };
        let task = task.trim();
        if task.is_empty() {
            continue;
        }
        if matches!(task, "/quit" | "/exit") {
            break;
        }
        match run_task(
            config,
            context,
            session_id,
            history.clone(),
            task.to_owned(),
        )
        .await
        {
            Ok(state) => history = state.messages,
            Err(CliError::Agent(AgentError::Cancelled)) => {
                eprintln!("task cancelled");
                break;
            }
            Err(error) => {
                eprintln!("{error}");
                if let Ok(database) = storage(config).await {
                    if let Ok(state) = database.load_latest(&session_id.to_string()).await {
                        history = state.messages;
                    }
                }
            }
        }
    }
    Ok(())
}

fn read_console_line() -> io::Result<Option<String>> {
    let mut bytes = Vec::new();
    let count = io::stdin().lock().read_until(b'\n', &mut bytes)?;
    if count == 0 {
        return Ok(None);
    }
    decode_console_input(&bytes).map(Some)
}

fn decode_console_input(bytes: &[u8]) -> io::Result<String> {
    if let Ok(text) = std::str::from_utf8(bytes) {
        return Ok(text.to_owned());
    }
    encoding_rs::EUC_KR
        .decode_without_bom_handling_and_without_replacement(bytes)
        .map(|text| text.into_owned())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "console input is neither UTF-8 nor Windows-949",
            )
        })
}

async fn run_task(
    config: &AppConfig,
    context: &ContextProfile,
    session_id: SessionId,
    history: Vec<Message>,
    task: String,
) -> Result<AgentState, CliError> {
    let database = Arc::new(storage(config).await?);
    let RunnerBundle { runner, mcp } = runner(config, context, database).await?;
    let result = runner.run_in_session(session_id, history, task).await;
    mcp.shutdown().await;
    result.map_err(CliError::from)
}

struct RunnerBundle {
    runner: AgentRunner,
    mcp: McpManager,
}

async fn runner(
    config: &AppConfig,
    context: &ContextProfile,
    database: Arc<SqliteSessionRepository>,
) -> Result<RunnerBundle, CliError> {
    tokio::fs::create_dir_all(&config.security.workspace_root).await?;
    let workspace = WorkspaceGuard::new(&config.security.workspace_root)?;
    let provider = Arc::new(provider(config)?);
    provider.health().await?;
    let (registry, mcp) = registry(config, &workspace, database.clone()).await?;
    let registry = Arc::new(registry);
    let jsonl = Arc::new(JsonlAuditSink::open(config.logging.directory.join("audit.jsonl")).await?);
    let audit_sinks: Vec<Arc<dyn agent_security::AuditSink>> = vec![database.clone(), jsonl];
    let audit = Arc::new(CompositeAuditSink::new(audit_sinks));
    let cancellation = CancellationToken::new();
    let signal = cancellation.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            signal.cancel();
        }
    });
    let command_profiles = command_profiles(config);
    let default_command = command_profiles.default.clone();
    let runner = AgentRunner::new(AgentRunnerConfig {
        provider,
        registry,
        approver: Arc::new(CliApprover {
            cancellation: cancellation.clone(),
        }),
        audit,
        events: Arc::new(CliEvents::new()),
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
            command_profiles,
        },
        sampling: SamplingConfig {
            temperature: config.model.sampling.temperature,
            top_p: config.model.sampling.top_p,
            top_k: config.model.sampling.top_k,
            repeat_penalty: config.model.sampling.repeat_penalty,
        },
        context_profile: context.clone(),
        system_prompt: SYSTEM_PROMPT.to_owned(),
        cancellation,
        sessions: Some(database),
    });
    Ok(RunnerBundle { runner, mcp })
}

async fn storage(config: &AppConfig) -> Result<SqliteSessionRepository, CliError> {
    SqliteSessionRepository::open(&config.storage.database_path)
        .await
        .map_err(CliError::Storage)
}

async fn documents(config: &AppConfig, command: DocumentCommand) -> Result<(), CliError> {
    tokio::fs::create_dir_all(&config.security.workspace_root).await?;
    let workspace = WorkspaceGuard::new(&config.security.workspace_root)?;
    let workspace_key = workspace.root().display().to_string();
    let repository = storage(config).await?;
    let service = DocumentService::new(config.documents.clone())
        .map_err(|e| CliError::Config(e.to_string()))?;
    match command {
        DocumentCommand::Add { paths, json } => {
            if paths.is_empty() {
                return Err(CliError::Config(
                    "at least one document path is required".into(),
                ));
            }
            let safe_inputs = paths
                .into_iter()
                .map(|path| workspace.resolve_existing(path))
                .collect::<Result<Vec<_>, _>>()?;
            let paths = collect_supported_paths(
                workspace.root(),
                &safe_inputs,
                service.limits().max_documents_per_request,
            )
            .map_err(|e| CliError::Storage(e.to_string()))?;
            let mut results = Vec::new();
            for path in paths {
                let resolved = workspace.resolve_existing(path)?;
                let relative = resolved
                    .strip_prefix(workspace.root())
                    .map_err(|e| CliError::Storage(e.to_string()))?
                    .to_string_lossy()
                    .replace('\\', "/");
                let bytes = tokio::fs::read(resolved).await?;
                let document = service
                    .parse(&workspace_key, &relative, &bytes)
                    .map_err(|e| CliError::Storage(e.to_string()))?;
                results.push(
                    repository
                        .upsert(&document)
                        .await
                        .map_err(|e| CliError::Storage(e.to_string()))?,
                );
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&results)
                        .map_err(|e| CliError::Storage(e.to_string()))?
                );
            } else {
                for result in results {
                    println!(
                        "{}\t{:?}\tchunks={}{}",
                        result.document.path,
                        result.document.status,
                        result.document.chunk_count,
                        if result.unchanged { "\tunchanged" } else { "" }
                    );
                }
            }
        }
        DocumentCommand::List {
            status,
            limit,
            json,
        } => {
            if limit == 0 {
                return Err(CliError::Config("limit must be positive".into()));
            }
            let values = repository
                .list(&workspace_key, status.map(Into::into), limit)
                .await
                .map_err(|e| CliError::Storage(e.to_string()))?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&values)
                        .map_err(|e| CliError::Storage(e.to_string()))?
                )
            } else {
                for value in values {
                    println!(
                        "{}\t{}\t{:?}\tchunks={}",
                        value.id, value.path, value.status, value.chunk_count
                    )
                }
            }
        }
        DocumentCommand::Show { id, chunks, json } => {
            let value = repository
                .get(&workspace_key, &id, chunks)
                .await
                .map_err(|e| CliError::Storage(e.to_string()))?
                .ok_or_else(|| CliError::Storage(format!("document not found: {id}")))?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&value)
                        .map_err(|e| CliError::Storage(e.to_string()))?
                )
            } else {
                println!(
                    "{}\npath: {}\nstatus: {:?}\nchunks: {}",
                    value.id,
                    value.source.path,
                    value.status,
                    value.chunks.len()
                );
                if chunks {
                    for chunk in value.chunks {
                        println!(
                            "\n{} page={:?} heading={:?} @{}-{}\n{}",
                            chunk.id,
                            chunk.page,
                            chunk.heading,
                            chunk.start_offset,
                            chunk.end_offset,
                            chunk.text
                        )
                    }
                }
            }
        }
        DocumentCommand::Search {
            query,
            documents,
            limit,
            json,
        } => {
            if query.trim().is_empty() {
                return Err(CliError::Config("query must not be empty".into()));
            }
            let limit = limit.unwrap_or(service.limits().default_search_limit);
            if limit == 0 || limit > service.limits().max_search_limit {
                return Err(CliError::Config(
                    "search limit is outside configured range".into(),
                ));
            }
            let values = repository
                .search(DocumentSearchQuery {
                    workspace: workspace_key,
                    query,
                    document_ids: documents,
                    limit,
                })
                .await
                .map_err(|e| CliError::Storage(e.to_string()))?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&values)
                        .map_err(|e| CliError::Storage(e.to_string()))?
                )
            } else {
                for value in values {
                    println!("{:.4}\t{}\n{}", value.score, value.citation, value.excerpt)
                }
            }
        }
    }
    Ok(())
}

async fn sessions(
    config: &AppConfig,
    context: &ContextProfile,
    command: SessionCommand,
) -> Result<(), CliError> {
    let database = Arc::new(storage(config).await?);
    match command {
        SessionCommand::List { limit } => {
            for session in database
                .list_sessions(limit)
                .await
                .map_err(CliError::Storage)?
            {
                println!(
                    "{}\t{}\t{}\t{}\t{}",
                    session.id,
                    session.status,
                    session.updated_at,
                    session.workspace,
                    session.recent_task
                );
            }
        }
        SessionCommand::Show {
            id,
            limit,
            all,
            json: json_output,
            research,
        } => {
            let limit = if all {
                None
            } else {
                Some(limit.unwrap_or(100))
            };
            let value = if research {
                database.show_research(&id, limit).await
            } else {
                database.show_session(&id, limit).await
            }
            .map_err(CliError::Storage)?;
            if json_output {
                println!(
                    "{}",
                    serde_json::to_string(&value)
                        .map_err(|error| CliError::Storage(error.to_string()))?
                );
            } else if research {
                print_research_summary(&value);
            } else {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&value)
                        .map_err(|error| CliError::Storage(error.to_string()))?
                );
            }
        }
        SessionCommand::Resume { id } => {
            let mut state = database.load_latest(&id).await.map_err(CliError::Storage)?;
            let canonical = std::fs::canonicalize(&config.security.workspace_root)?;
            if state.workspace != canonical.display().to_string() {
                return Err(CliError::Storage(format!(
                    "session workspace {} does not match configured workspace {}",
                    state.workspace,
                    canonical.display()
                )));
            }
            if !matches!(
                state.status,
                agent_core::AgentStatus::Completed
                    | agent_core::AgentStatus::Failed
                    | agent_core::AgentStatus::Cancelled
            ) {
                let RunnerBundle { runner, mcp } =
                    runner(config, context, database.clone()).await?;
                let resumed = runner.resume(state).await;
                mcp.shutdown().await;
                let value = resumed?;
                state = value;
            }
            chat_session(config, context, state.session_id, state.messages).await?;
        }
        SessionCommand::Prune { older_than, yes } => {
            if older_than < 0 {
                return Err(CliError::Config(
                    "--older-than must be non-negative".to_owned(),
                ));
            }
            let count = database
                .prune_count(older_than)
                .await
                .map_err(CliError::Storage)?;
            if count == 0 {
                println!("no terminal sessions matched");
                return Ok(());
            }
            let confirmed = if yes {
                true
            } else {
                print!("delete {count} terminal sessions from SQLite? [y/N] ");
                io::stdout().flush()?;
                read_console_line()?.is_some_and(|answer| {
                    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
                })
            };
            if confirmed {
                let deleted = database
                    .prune(older_than)
                    .await
                    .map_err(CliError::Storage)?;
                println!("deleted {deleted} sessions; logs/audit.jsonl remains append-only");
            } else {
                println!("prune cancelled");
            }
        }
    }
    Ok(())
}

fn print_research_summary(value: &serde_json::Value) {
    println!(
        "session={} status={} updated_at={}",
        value["session"]["id"].as_str().unwrap_or("-"),
        value["session"]["status"].as_str().unwrap_or("-"),
        value["session"]["updated_at"].as_str().unwrap_or("-")
    );
    let Some(entries) = value["research"].as_array() else {
        return;
    };
    for entry in entries {
        let updated_at = entry["updated_at"].as_str().unwrap_or("-");
        let status = entry["status"].as_str().unwrap_or("-");
        if entry["kind"] == "web_search" {
            let query = entry["query"].as_str().unwrap_or("-");
            if entry["skipped_duplicate"] == true {
                println!("[search] {updated_at} status={status} skipped_duplicate query={query:?}");
            } else {
                let count = entry["result_count"].as_u64().unwrap_or(0);
                let provider = entry["provider"].as_str().unwrap_or("-");
                println!(
                    "[search] {updated_at} status={status} provider={provider} results={count} query={query:?}"
                );
                if let Some(sources) = entry["sources"].as_array() {
                    for source in sources {
                        println!(
                            "  {}. {} — {}",
                            source["rank"].as_u64().unwrap_or(0),
                            source["title"].as_str().unwrap_or("(untitled)"),
                            source["url"].as_str().unwrap_or("-")
                        );
                    }
                }
            }
        } else {
            println!(
                "[fetch] {updated_at} status={status} requested={} final={} fetched_at={} content_type={} bytes={}{}",
                entry["requested_url"].as_str().unwrap_or("-"),
                entry["final_url"].as_str().unwrap_or("-"),
                entry["fetched_at"].as_str().unwrap_or("-"),
                entry["content_type"].as_str().unwrap_or("-"),
                entry["received_bytes"].as_u64().unwrap_or(0),
                entry["error"]
                    .as_str()
                    .map(|error| format!(" error={error:?}"))
                    .unwrap_or_default()
            );
        }
    }
}

fn command_profiles(config: &AppConfig) -> CommandProfiles {
    let defaults = CommandProfiles::default();
    let flat = CommandLimits {
        timeout_seconds: config.tools.command_timeout_seconds,
        stdout_limit_bytes: config.tools.stdout_limit_bytes,
        stderr_limit_bytes: config.tools.stderr_limit_bytes,
    };
    CommandProfiles {
        default: apply_command_overrides(flat, &config.tools.command_profiles.default),
        cargo_build: apply_command_overrides(
            defaults.cargo_build,
            &config.tools.command_profiles.cargo_build,
        ),
        cargo_test: apply_command_overrides(
            defaults.cargo_test,
            &config.tools.command_profiles.cargo_test,
        ),
        git: apply_command_overrides(defaults.git, &config.tools.command_profiles.git),
    }
}

fn apply_command_overrides(
    mut limits: CommandLimits,
    overrides: &CommandLimitOverrides,
) -> CommandLimits {
    if let Some(value) = overrides.timeout_seconds {
        limits.timeout_seconds = value;
    }
    if let Some(value) = overrides.stdout_limit_bytes {
        limits.stdout_limit_bytes = value;
    }
    if let Some(value) = overrides.stderr_limit_bytes {
        limits.stderr_limit_bytes = value;
    }
    limits
}

async fn registry(
    config: &AppConfig,
    workspace: &WorkspaceGuard,
    database: Arc<SqliteSessionRepository>,
) -> Result<(ToolRegistry, McpManager), CliError> {
    let mut value = ToolRegistry::new();
    register_builtin_tools(&mut value)?;
    let timeout = Duration::from_secs(config.research.request_timeout_seconds);
    let provider = Arc::new(
        SearxngProvider::new(
            &config.research.searxng_base_url,
            timeout,
            &config.research.user_agent,
        )
        .map_err(|error| CliError::Config(error.to_string()))?,
    );
    let fetcher = HttpFetcher::new(FetchPolicy::production(
        timeout,
        config.research.max_redirects,
        config.research.max_response_bytes,
        config.research.user_agent.clone(),
    ))
    .map_err(|error| CliError::Config(error.to_string()))?;
    register_research_tools(&mut value, provider, fetcher, config.research.max_results)?;
    let document_service = DocumentService::new(config.documents.clone())
        .map_err(|error| CliError::Config(error.to_string()))?;
    register_document_tools(&mut value, database, document_service)?;
    let connected = McpManager::connect_enabled(&config.mcp, workspace).await?;
    for diagnostic in &connected.diagnostics {
        eprintln!("[mcp:{}] {}", diagnostic.server, diagnostic.message);
    }
    for tool in connected.tools {
        let name = tool.definition().function.name;
        if let Err(error) = value.register_arc(tool) {
            eprintln!("[mcp:{name}] {error}; Tool skipped");
        }
    }
    Ok((value, connected.manager))
}

fn provider(config: &AppConfig) -> Result<OpenAiCompatibleProvider, agent_model::ModelError> {
    OpenAiCompatibleProvider::new(
        &config.model.base_url,
        &config.model.model,
        Duration::from_secs(config.model.request_timeout_seconds),
    )
}

fn init_logging(config: &AppConfig) -> Result<WorkerGuard, CliError> {
    std::fs::create_dir_all(&config.logging.directory)?;
    let appender = tracing_appender::rolling::daily(&config.logging.directory, "agent.log");
    let (writer, guard) = tracing_appender::non_blocking(appender);
    let filter = tracing_subscriber::EnvFilter::try_new(&config.logging.level)
        .map_err(|e| CliError::Config(e.to_string()))?;
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_writer(io::stderr))
        .with(tracing_subscriber::fmt::layer().json().with_writer(writer))
        .try_init()
        .map_err(|e| CliError::Config(e.to_string()))?;
    Ok(guard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn research_session_view_flag_parses() -> Result<(), Box<dyn std::error::Error>> {
        let cli = Cli::try_parse_from([
            "veyra",
            "sessions",
            "show",
            "session-id",
            "--research",
            "--json",
        ])?;
        assert!(matches!(
            cli.command,
            Some(Commands::Sessions {
                command: SessionCommand::Show {
                    research: true,
                    json: true,
                    ..
                }
            })
        ));
        Ok(())
    }

    #[test]
    fn web_search_limit_has_a_specific_cli_qualifier() {
        let result = agent_tools::ToolResult {
            content: serde_json::Value::Null,
            summary: "search".to_owned(),
            truncated: true,
            metadata: serde_json::json!({"kind":"web_search","limit_reached":true}),
        };
        assert_eq!(
            tool_result_qualifier(&result),
            " (configured limit reached)"
        );
    }

    #[test]
    fn streaming_state_closes_an_open_line_without_losing_response_state() {
        let events = CliEvents::new();
        events.record_token("partial answer");
        assert!(events.stream_boundary_pending.load(Ordering::Relaxed));
        assert!(events.streamed_current_response.load(Ordering::Relaxed));
        assert!(events.close_stream_line());
        assert!(!events.stream_boundary_pending.load(Ordering::Relaxed));
        assert!(events.streamed_current_response.load(Ordering::Relaxed));
    }

    #[test]
    fn invalid_limits_are_rejected() {
        let mut config = AppConfig::default();
        config.agent.max_iterations = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn invalid_research_configuration_is_rejected() {
        let mut config = AppConfig::default();
        config.research.max_results = 0;
        assert!(config.validate().is_err());
        config = AppConfig::default();
        config.research.searxng_base_url = "file:///tmp/search".to_owned();
        assert!(config.validate().is_err());
    }

    #[tokio::test]
    async fn registry_includes_research_and_document_tools() -> Result<(), CliError> {
        let temp = tempfile::tempdir()?;
        let workspace = WorkspaceGuard::new(temp.path())?;
        let database = Arc::new(
            SqliteSessionRepository::open(temp.path().join("test.db"))
                .await
                .map_err(CliError::Storage)?,
        );
        let (registry, manager) = registry(&AppConfig::default(), &workspace, database).await?;
        let definitions = registry.definitions();
        assert!(
            definitions
                .iter()
                .any(|definition| definition.function.name == "web_search")
        );
        for name in ["document_index", "document_list", "document_search"] {
            assert!(
                definitions
                    .iter()
                    .any(|definition| definition.function.name == name)
            );
        }
        assert!(
            definitions
                .iter()
                .any(|definition| definition.function.name == "http_fetch")
        );
        manager.shutdown().await;
        Ok(())
    }

    #[test]
    fn context_profile_precedence_and_legacy_compatibility() {
        let mut config = AppConfig::default();
        assert_eq!(
            config.resolved_context_profile(None).name,
            agent_context::ContextProfileName::Default
        );
        config.model.context_size = 16_384;
        assert_eq!(
            config.resolved_context_profile(None).name,
            agent_context::ContextProfileName::Legacy
        );
        config.context = Some(ContextSection {
            profile: ContextProfileArg::Default,
        });
        assert_eq!(
            config
                .resolved_context_profile(Some(ContextProfileArg::Large))
                .budget
                .context_limit,
            65_536
        );
    }

    #[test]
    fn embedded_system_prompt_requires_korean_responses() {
        assert!(SYSTEM_PROMPT.contains("Write all assistant prose in Korean"));
        assert!(SYSTEM_PROMPT.contains("switch languages"));
    }

    #[test]
    fn console_input_accepts_utf8_korean() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            decode_console_input("한국어 입력\n".as_bytes())?,
            "한국어 입력\n"
        );
        Ok(())
    }

    #[test]
    fn console_input_accepts_windows_949_korean() -> Result<(), Box<dyn std::error::Error>> {
        let (bytes, _, had_errors) = encoding_rs::EUC_KR.encode("한국어 입력\n");
        assert!(!had_errors);
        assert_eq!(decode_console_input(&bytes)?, "한국어 입력\n");
        Ok(())
    }

    #[tokio::test]
    async fn approval_input_observes_cancellation_without_a_line() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let (_sender, receiver) = tokio::sync::oneshot::channel();
        assert!(matches!(
            wait_for_approval_input(&cancellation, receiver).await,
            ApprovalInput::Cancelled
        ));
    }
}
