use miniqwen_core::{ChatMessage, GenerateParams, InferencePipeline, StreamAction};
use anyhow::Result;
use clap::Parser;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "MiniQwen CLI - local multimodal inference")]
struct Args {
    /// User prompt text
    #[arg(short, long)]
    prompt: String,

    /// Path to an image file (enables multimodal mode)
    #[arg(short, long)]
    image: Option<PathBuf>,

    /// Path to the model directory
    #[arg(short, long, default_value = "models")]
    model_dir: PathBuf,

    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 200)]
    max_tokens: usize,

    /// Enable thinking / chain-of-thought mode
    #[arg(long, default_value_t = false)]
    enable_thinking: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.model_dir.exists() {
        eprintln!("Error: Model directory {:?} not found", args.model_dir);
        std::process::exit(1);
    }

    println!("Loading model from {:?}...", args.model_dir);
    let mut pipeline = InferencePipeline::load(&args.model_dir)?;

    // Build message: if image is provided, use multimodal message
    let messages = if args.image.is_some() {
        vec![ChatMessage::user_with_image(&args.prompt)]
    } else {
        vec![ChatMessage::user(&args.prompt)]
    };

    // Load image if provided
    let image = if let Some(ref image_path) = args.image {
        if !image_path.exists() {
            eprintln!("Error: Image file {:?} not found", image_path);
            std::process::exit(1);
        }
        println!("Loading image {:?}...", image_path);
        Some(image::open(image_path)?)
    } else {
        None
    };

    let params = GenerateParams {
        max_tokens: args.max_tokens,
        enable_thinking: args.enable_thinking,
    };

    println!("\n--- Generating ---\n");

    let _output = pipeline.generate(
        &messages,
        None,
        image.as_ref(),
        &params,
        |token_text| {
            print!("{}", token_text);
            io::stdout().flush().ok();
            StreamAction::Continue
        },
    )?;

    println!("\n\n--- Done ---");
    Ok(())
}
