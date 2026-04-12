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

#[allow(dead_code)]
#[derive(Deserialize, Debug, Clone)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize, // Matches JSON
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub num_heads: usize,
    pub spatial_merge_size: usize,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug, Clone)]
pub struct QwenConfig {
    pub model_type: String,
    pub text_config: TextConfig,
    pub vision_config: Option<VisionConfig>,
    pub image_token_id: Option<u32>,
    pub vision_start_token_id: Option<u32>,
    pub vision_end_token_id: Option<u32>,
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
            Some(serde_json::Value::Array(a)) => a.first().and_then(|v| v.as_u64()).unwrap_or(151643) as u32,
            _ => 151643,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_deserialization() {
        let json = r#"{
            "model_type": "qwen2_vl",
            "text_config": {
                "num_hidden_layers": 2,
                "num_key_value_heads": 4,
                "hidden_size": 128,
                "vocab_size": 1000,
                "eos_token_id": 151643,
                "num_attention_heads": 8,
                "layer_types": ["full_attention", "full_attention"],
                "linear_num_key_heads": 0,
                "linear_key_head_dim": 0,
                "linear_num_value_heads": 0,
                "linear_value_head_dim": 0,
                "linear_conv_kernel_dim": 0
            },
            "vision_config": {
                "depth": 1,
                "hidden_size": 64,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "num_heads": 4,
                "spatial_merge_size": 2
            }
        }"#;
        let config: QwenConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, "qwen2_vl");
        assert_eq!(config.num_hidden_layers(), 2);
        assert_eq!(config.head_dim(), 16); // 128 / 8
        assert_eq!(config.get_eos_id(), 151643);
    }

    #[test]
    fn test_eos_id_array() {
        let json = r#"{
            "model_type": "qwen2",
            "text_config": {
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "hidden_size": 64,
                "vocab_size": 100,
                "eos_token_id": [123, 456],
                "num_attention_heads": 1,
                "layer_types": ["full_attention"],
                "linear_num_key_heads": 0,
                "linear_key_head_dim": 0,
                "linear_num_value_heads": 0,
                "linear_value_head_dim": 0,
                "linear_conv_kernel_dim": 0
            }
        }"#;
        let config: QwenConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.get_eos_id(), 123);
    }
}
