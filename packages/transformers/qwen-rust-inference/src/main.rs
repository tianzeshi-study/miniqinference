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
    /// 包含 config.json, tokenizer.json 以及 onnx/ 目录的路径
    #[arg(short, long, default_value = "models")]
    model_dir: PathBuf,
    #[arg(short, long, default_value = "你好，请自我介绍一下。")]
    prompt: String,
    #[arg(long, default_value_t = 200)]
    max_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 1. 验证并加载配置
    if !args.model_dir.exists() {
        println!("错误: 找不到目录 {:?}", args.model_dir);
        return Ok(());
    }

    println!("正在加载配置...");
    let config = QwenConfig::from_file(args.model_dir.join("config.json"))?;
    
    println!("正在加载分词器...");
    let tokenizer = Tokenizer::from_file(args.model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("分词器加载失败: {}", e))?;

    println!("正在加载 ONNX 模型 (包括 Embedding 和 Merged Decoder)...");
    let mut engine = QwenEngine::new(&args.model_dir, config.clone())?;

    // 2. 预处理输入
    let encoding = tokenizer.encode(args.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("分词失败: {}", e))?;
    let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let mut position_ids: Vec<i64> = (0..input_ids.len() as i64).collect();
    let mut kv_cache: Vec<DynValue> = engine.create_empty_kv_cache()?;

    println!("\n生成中:\n---");
    print!("{}", args.prompt);
    io::stdout().flush()?;

    // 3. 自回归循环
    let mut is_prefill = true;
    let eos_id = config.get_eos_id();

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
            .map_err(|e| anyhow::anyhow!("解码失败: {}", e))?;
        print!("{}", output_text);
        io::stdout().flush()?;

        // 状态更新：
        // 1. 切换到解码模式 (is_prefill = false)
        // 2. 下一轮只输入最新的 token
        // 3. position_id 递增
        is_prefill = false;
        input_ids = vec![next_token as i64];
        let next_pos = position_ids.last().unwrap_or(&-1) + 1;
        position_ids = vec![next_pos]; 
    }

    println!("\n---\n生成结束。");
    Ok(())
}
