mod api;
mod types;

use api::AppState;
use axum::{routing::{get, post}, Router};
use clap::Parser;
use miniqwen_core::InferencePipeline;
use std::{path::PathBuf, sync::Arc};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

#[derive(Parser, Debug)]
#[command(author, version, about = "MiniQwen Server - OpenAI-compatible inference API")]
struct Args {
    /// Path to the model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Host address to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 8000)]
    port: u16,

    /// Model name to expose in API
    #[arg(long)]
    model_name: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    if !args.model_dir.exists() {
        eprintln!("Error: Model directory {:?} not found", args.model_dir);
        std::process::exit(1);
    }

    // Derive model name from directory if not specified
    let model_name = args.model_name.unwrap_or_else(|| {
        args.model_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "qwen".to_string())
    });

    tracing::info!("Loading model from {:?}...", args.model_dir);
    let pipeline = InferencePipeline::load(&args.model_dir)?;
    tracing::info!("Model loaded successfully as '{}'", model_name);

    let state = Arc::new(AppState {
        pipeline: Mutex::new(pipeline),
        model_name,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(api::chat_completions))
        .route("/v1/models", get(api::list_models))
        .route("/health", get(|| async { "ok" }))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Server listening on http://{}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}
