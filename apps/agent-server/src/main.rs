use agent_app::{AppConfig, ApplicationService};
use agent_server::{router, validate_bind};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "agent-server",
    version,
    about = "Veyra local Agent API and Web UI"
)]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    workspace: Option<PathBuf>,
    #[arg(long)]
    bind: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let args = Args::parse();
    let mut config = AppConfig::load(args.config.as_deref(), args.workspace)?;
    if let Some(bind) = args.bind {
        config.server.bind = bind;
    }
    let token = std::env::var("VEYRA_SERVER_TOKEN").ok();
    let (address, token) = validate_bind(&config.server.bind, config.server.allow_remote, token)?;
    if !address.ip().is_loopback() {
        tracing::warn!(%address, "Veyra API is remotely exposed with bearer authentication");
    }
    let frontend = config.server.frontend_directory.clone();
    let service = ApplicationService::open(config).await?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    tracing::info!(%address, "Veyra v0.9 server ready");
    axum::serve(listener, router(service, token, frontend)).await?;
    Ok(())
}
