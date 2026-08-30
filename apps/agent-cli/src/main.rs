use agent_core::{AgentEvent, AgentEventSink, AgentLimits, AgentRunner, AgentRunnerConfig};
use agent_model::{ModelProvider, OpenAiCompatibleProvider, SamplingConfig};
use agent_security::{
    ApprovalDecision, ApprovalProvider, ApprovalRequest, JsonlAuditSink, WorkspaceGuard,
};
use agent_tools::{
    CommandLimits, CommandProfiles, ExecutionLimits, ToolRegistry, register_builtin_tools,
};
use async_trait::async_trait;
use chrono::Utc;
use clap::{Parser, Subcommand};
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

#[derive(Parser)]
#[command(name = "veyra", version, about = "A safe local coding agent")]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    workspace: Option<PathBuf>,
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
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct AppConfig {
    agent: AgentSection,
    model: ModelSection,
    security: SecuritySection,
    tools: ToolsSection,
    logging: LoggingSection,
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
        Ok(())
    }
}

struct CliApprover;
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
        let input = read_console_line().ok().flatten().unwrap_or_default();
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
    streamed: AtomicBool,
}
impl CliEvents {
    fn new() -> Self {
        Self {
            streamed: AtomicBool::new(false),
        }
    }
}
#[async_trait]
impl AgentEventSink for CliEvents {
    async fn emit(&self, event: AgentEvent) {
        match event {
            AgentEvent::TokenDelta { text, .. } => {
                self.streamed.store(true, Ordering::Relaxed);
                print!("{text}");
                let _ = io::stdout().flush();
            }
            AgentEvent::ToolStarted { call_id } => eprintln!("\n[tool {call_id}] started"),
            AgentEvent::ToolCompleted { result, .. } => eprintln!(
                "[tool] {}{}",
                result.summary,
                if result.truncated { " (truncated)" } else { "" }
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
            AgentEvent::TaskCompleted { answer, .. } => {
                if self.streamed.swap(false, Ordering::Relaxed) {
                    println!();
                } else {
                    println!("{answer}");
                }
            }
            AgentEvent::TaskFailed { error, .. } => eprintln!("task failed: {error}"),
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), CliError> {
    let cli = Cli::parse();
    let config = AppConfig::load(cli.config.as_deref(), cli.workspace)?;
    let _log_guard = init_logging(&config)?;
    match cli.command.unwrap_or(Commands::Chat) {
        Commands::Config {
            command: ConfigCommand::Check,
        } => {
            println!("configuration is valid");
            Ok(())
        }
        Commands::Tools {
            command: ToolCommand::List,
        } => {
            let registry = registry()?;
            for definition in registry.definitions() {
                println!(
                    "{}\t{}",
                    definition.function.name, definition.function.description
                );
            }
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
        Commands::Run { task } => run_task(&config, task).await,
        Commands::Chat => chat(&config).await,
    }
}

async fn chat(config: &AppConfig) -> Result<(), CliError> {
    println!("Veyra v0.2 — enter a task, or /quit to exit.");
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
        if let Err(error) = run_task(config, task.to_owned()).await {
            eprintln!("{error}");
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

async fn run_task(config: &AppConfig, task: String) -> Result<(), CliError> {
    tokio::fs::create_dir_all(&config.security.workspace_root).await?;
    let workspace = WorkspaceGuard::new(&config.security.workspace_root)?;
    let provider = Arc::new(provider(config)?);
    provider.health().await?;
    let registry = Arc::new(registry()?);
    let audit = Arc::new(JsonlAuditSink::open(config.logging.directory.join("audit.jsonl")).await?);
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
        approver: Arc::new(CliApprover),
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
        system_prompt: include_str!("../../../prompts/system.md").to_owned(),
        cancellation,
    });
    runner.run(task).await?;
    Ok(())
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

fn registry() -> Result<ToolRegistry, agent_tools::ToolError> {
    let mut value = ToolRegistry::new();
    register_builtin_tools(&mut value)?;
    Ok(value)
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
    fn invalid_limits_are_rejected() {
        let mut config = AppConfig::default();
        config.agent.max_iterations = 0;
        assert!(config.validate().is_err());
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
}
