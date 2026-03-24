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
    #[arg(short, long, default_value = "models")]
    model_dir: PathBuf,
    #[arg(short, long, default_value = "你好，请自我介绍一下。")]
    prompt: String,
    #[arg(long, default_value_t = 200)]
    max_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.model_dir.exists() {
        println!("Error: Directory {:?} not found", args.model_dir);
        return Ok(());
    }

    println!("Loading configuration...");
    let config = QwenConfig::from_file(args.model_dir.join("config.json"))?;
    
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(args.model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    println!("Loading ONNX models...");
    let mut engine = QwenEngine::new(&args.model_dir, config.clone())?;

    let encoding = tokenizer.encode(args.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let mut position_ids: Vec<i64> = (0..input_ids.len() as i64).collect();
    let mut kv_cache: Vec<DynValue> = engine.create_empty_kv_cache()?;

    println!("\nGenerating:\n---");
    print!("{}", args.prompt);
    io::stdout().flush()?;

    let mut is_prefill = true;
    let eos_id = config.get_eos_id();
    let mut current_pos = input_ids.len() as i64;

    for _ in 0..args.max_tokens {
        let next_token = engine.forward(
            input_ids.clone(), 
            &mut kv_cache, 
            position_ids.clone(),
            is_prefill
        )?;
        
        if next_token == eos_id {
            break;
        }

        let output_text = tokenizer.decode(&[next_token], true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
        print!("{}", output_text);
        io::stdout().flush()?;

        is_prefill = false;
        input_ids = vec![next_token as i64];
        position_ids = vec![current_pos];
        current_pos += 1;
    }

    println!("\n---\nDone.");
    Ok(())
}
