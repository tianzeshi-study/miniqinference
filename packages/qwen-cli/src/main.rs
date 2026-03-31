use qwen_core::config::QwenConfig;
use qwen_core::model::QwenEngine;
use anyhow::Result;
use clap::Parser;
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
    #[arg(short, long)]
    image: Option<PathBuf>,
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

    let final_prompt = if args.image.is_some() {
        // Construct visual prompt if image is provided
        format!("<|vision_start|><|image_pad|><|vision_end|>{}", args.prompt)
    } else {
        args.prompt.clone()
    };

    let encoding = tokenizer.encode(final_prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    
    let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let mut kv_cache: Vec<DynValue> = engine.create_empty_kv_cache()?;

    println!("\nGenerating:\n---");
    if args.image.is_some() {
        print!("[Image] ");
    }
    print!("{}", args.prompt);
    io::stdout().flush()?;

    let eos_id = config.get_eos_id();
    
    // Find image token index (placeholder)
    let image_pad_token_id = config.image_token_id.unwrap_or(248056) as i64;
    let image_token_indices: Vec<usize> = input_ids.iter().enumerate()
        .filter(|(_, &id)| id == image_pad_token_id)
        .map(|(i, _)| i)
        .collect();

    let mut next_token: u32;
    let mut current_pos: i64;

    if let (Some(image_path), true) = (args.image, !image_token_indices.is_empty()) {
        println!("\n[Vision] Encoding image {:?}...", image_path);
        let img = image::open(image_path)?;
        let (image_embeds, _shape, _thw) = engine.encode_image(&img)?;
        
        println!("[Vision] Running prefill with image...");
        next_token = engine.forward_with_vision(
            input_ids.clone(),
            image_embeds,
            // shape,
            image_token_indices,
            &mut kv_cache,
        )?;
        current_pos = input_ids.len() as i64; // This is a bit complex due to padding replacement
        // Note: the effective seq_len in forward_with_vision is longer
        // But for position_ids in next steps, we need to track it.
        // Simplified: let's re-calculate it in main or trust engine.
    } else {
        next_token = engine.forward(
            input_ids.clone(),
            &mut kv_cache,
            (0..input_ids.len() as i64).collect(),
            true
        )?;
        current_pos = input_ids.len() as i64;
    }

    // Decoding loop
    for _ in 0..args.max_tokens {
        if next_token == eos_id {
            break;
        }

        let output_text = tokenizer.decode(&[next_token], true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
        print!("{}", output_text);
        io::stdout().flush()?;

        input_ids = vec![next_token as i64];
        next_token = engine.forward(
            input_ids.clone(),
            &mut kv_cache,
            vec![current_pos],
            false
        )?;
        current_pos += 1;
    }

    println!("\n---\nDone.");
    Ok(())
}
