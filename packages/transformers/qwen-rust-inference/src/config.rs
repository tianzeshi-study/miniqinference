use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Deserialize, Debug, Clone)]
pub struct TextConfig {
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub eos_token_id: Option<serde_json::Value>,
    pub num_attention_heads: usize,
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub head_dim: Option<usize>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct QwenConfig {
    pub model_type: String,
    pub text_config: TextConfig,
}

impl QwenConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config = serde_json::from_reader(reader)?;
        Ok(config)
    }

    pub fn num_hidden_layers(&self) -> usize {
        self.text_config.num_hidden_layers
    }

    pub fn layer_type(&self, index: usize) -> &str {
        &self.text_config.layer_types[index]
    }

    pub fn head_dim(&self) -> usize {
        self.text_config.head_dim.unwrap_or_else(|| {
            self.text_config.hidden_size / self.text_config.num_attention_heads
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.text_config.vocab_size
    }

    pub fn get_eos_id(&self) -> u32 {
        match &self.text_config.eos_token_id {
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(151643) as u32,
            Some(serde_json::Value::Array(a)) => a.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32,
            _ => 151643,
        }
    }
}
