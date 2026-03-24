mod config;
mod model;

use anyhow::Result;
use clap::Parser;
use config::QwenConfig;
use model::QwenEngine;
use ort::value::DynValue;
use std::io::{self, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen Rust MVP Inference")]
struct Args {
    #[arg(short, long, default_value = "model.onnx")]
    model: PathBuf,
    #[arg(short, long, default_value = "tokenizer.json")]
    tokenizer: PathBuf,
    #[arg(short, long, default_value = "config.json")]
    config: PathBuf,
    #[arg(short, long, default_value = "你好，请问你是谁？")]
    prompt: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.config.exists() || !args.tokenizer.exists() || !args.model.exists() {
        println!("Error: Missing required files (config.json, tokenizer.json, or model.onnx).");
        println!("Please place them in the project root or specify paths via arguments.");
        return Ok(());
    }

    println!("Loading configuration...");
    let config = QwenConfig::from_file(&args.config)?;
    
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

    println!("Loading model (this may take a while)...");
    let mut engine = QwenEngine::new(&args.model, config.clone())?;

    let encoding = tokenizer.encode(args.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Encoding error: {}", e))?;
    let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let mut position_ids: Vec<i64> = (0..input_ids.len() as i64).collect();
    let mut kv_cache: Vec<DynValue> = engine.create_empty_kv_cache()?;

    println!("\nGenerating:\n---");
    print!("{}", args.prompt);
    io::stdout().flush()?;

    let eos_id = config.eos_token_id.unwrap_or(151643);

    for _ in 0..200 {
        let next_token = engine.forward(input_ids.clone(), &mut kv_cache, position_ids.clone())?;
        
        if next_token == eos_id {
            break;
        }

        let output_text = tokenizer.decode(&[next_token], true)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;
        print!("{}", output_text);
        io::stdout().flush()?;

        input_ids = vec![next_token as i64];
        let next_pos = position_ids.last().unwrap() + 1;
        position_ids = vec![next_pos]; 
    }

    println!("\n---\nDone.");
    Ok(())
}
