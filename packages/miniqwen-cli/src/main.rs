use miniqwen_core::{ChatMessage, GenerateParams, InferencePipeline, StreamAction};
use anyhow::Result;
use clap::Parser;
use std::io::{self, Write};
use std::path::PathBuf;
use tracing::debug;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[derive(Parser, Debug)]
#[command(author, version, about = "MiniQwen CLI - local multimodal inference")]
struct Args {
    /// User prompt text
    #[arg(short, long)]
    prompt: String,

    /// Path to an image file (enables multimodal mode)
    #[arg(short, long)]
    image: Option<PathBuf>,

    /// Base64 encoded image (enables multimodal mode)
    #[arg(short = 'b', long)]
    base64_image: Option<String>,

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
    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(io::stderr))
        .with(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    if !args.model_dir.exists() {
        eprintln!("Error: Model directory {:?} not found", args.model_dir);
        std::process::exit(1);
    }

    debug!("Loading model from {:?}...", args.model_dir);
    let mut pipeline = InferencePipeline::load(&args.model_dir)?;

    // Build message: if image is provided, use multimodal message
    let messages = if args.image.is_some() || args.base64_image.is_some() {
        vec![ChatMessage::user_with_image(&args.prompt)]
    } else {
        vec![ChatMessage::user(&args.prompt)]
    };

    // Load image if provided via file or base64
    let image = if let Some(ref image_path) = args.image {
        if !image_path.exists() {
            eprintln!("Error: Image file {:?} not found", image_path);
            std::process::exit(1);
        }
        debug!("Loading image {:?}...", image_path);
        Some(image::open(image_path)?)
    } else if let Some(ref b64_str) = args.base64_image {
        debug!("Decoding base64 image...");
        let b64_data = if b64_str.starts_with("data:image") {
            b64_str.splitn(2, ",").nth(1).ok_or_else(|| anyhow::anyhow!("Invalid data URL format"))?
        } else {
            b64_str
        };
        let decoded = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            b64_data,
        )?;
        Some(image::load_from_memory(&decoded)?)
    } else {
        None
    };

    let params = GenerateParams {
        max_tokens: args.max_tokens,
        enable_thinking: args.enable_thinking,
    };

    debug!("--- Generating ---");

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

    debug!("--- Done ---");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_args_parsing_image() {
        let args = Args::try_parse_from(&["miniqwen-cli", "--prompt", "hello", "--image", "test.jpg"]).unwrap();
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.image, Some(PathBuf::from("test.jpg")));
        assert_eq!(args.base64_image, None);
    }

    #[test]
    fn test_args_parsing_base64() {
        let args = Args::try_parse_from(&["miniqwen-cli", "--prompt", "hello", "-b", "YmFzZTY0"]).unwrap();
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.image, None);
        assert_eq!(args.base64_image, Some("YmFzZTY0".to_string()));
    }

    #[test]
    fn test_args_parsing_both() {
        let args = Args::try_parse_from(&["miniqwen-cli", "--prompt", "hello", "-i", "test.jpg", "-b", "YmFzZTY0"]).unwrap();
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.image, Some(PathBuf::from("test.jpg")));
        assert_eq!(args.base64_image, Some("YmFzZTY0".to_string()));
    }
}
