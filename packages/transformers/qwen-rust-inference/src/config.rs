use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Deserialize, Debug, Clone)]
pub struct QwenConfig {
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub eos_token_id: Option<u32>,
    pub num_attention_heads: usize,
}

impl QwenConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config = serde_json::from_reader(reader)?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}
