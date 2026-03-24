use crate::config::QwenConfig;
use anyhow::Result;
use ort::session::Session;
use ort::value::{DynValue, Value};
use std::path::Path;

pub struct QwenEngine {
    session: Session,
    config: QwenConfig,
}

impl QwenEngine {
    pub fn new(model_path: &Path, config: QwenConfig) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create builder: {:?}", e))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("Failed to set threads: {:?}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {:?}", e))?;
        Ok(Self { session, config })
    }

    pub fn forward(
        &mut self,
        input_ids: Vec<i64>,
        past_key_values: &mut Vec<DynValue>,
        position_ids: Vec<i64>,
    ) -> Result<u32> {
        let seq_len = input_ids.len();
        let batch_size = 1;

        let input_tensor = Value::from_array((vec![batch_size, seq_len], input_ids))
            .map_err(|e| anyhow::anyhow!("Failed to create input_ids: {:?}", e))?;
        let attention_mask = Value::from_array((vec![batch_size, position_ids.len()], vec![1i64; position_ids.len()]))
            .map_err(|e| anyhow::anyhow!("Failed to create attention_mask: {:?}", e))?;
        let position_tensor = Value::from_array((vec![batch_size, seq_len], position_ids))
            .map_err(|e| anyhow::anyhow!("Failed to create position_ids: {:?}", e))?;

        let mut inputs_map = vec![
            ("input_ids".to_string(), input_tensor.into_dyn()),
            ("attention_mask".to_string(), attention_mask.into_dyn()),
            ("position_ids".to_string(), position_tensor.into_dyn()),
        ];

        for i in 0..self.config.num_hidden_layers {
            let k = past_key_values.remove(0);
            let v = past_key_values.remove(0);
            inputs_map.push((format!("past_key_values.{}.key", i), k));
            inputs_map.push((format!("past_key_values.{}.value", i), v));
        }

        let mut outputs = self.session.run(inputs_map)
            .map_err(|e| anyhow::anyhow!("Inference failed: {:?}", e))?;

        let logits_value = outputs.remove("logits")
            .ok_or_else(|| anyhow::anyhow!("Missing logits"))?;
        
        let (_, logits_data) = logits_value.try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract logits: {:?}", e))?;
        
        let vocab_size = self.config.vocab_size;
        let last_token_logits = &logits_data[logits_data.len() - vocab_size..];
        
        let next_token = last_token_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();

        for i in 0..self.config.num_hidden_layers {
            let present_k = outputs.remove(format!("present.{}.key", i))
                .ok_or_else(|| anyhow::anyhow!("Missing present key"))?;
            let present_v = outputs.remove(format!("present.{}.value", i))
                .ok_or_else(|| anyhow::anyhow!("Missing present value"))?;
            past_key_values.push(present_k);
            past_key_values.push(present_v);
        }

        Ok(next_token)
    }

    pub fn create_empty_kv_cache(&self) -> Result<Vec<DynValue>> {
        let mut cache = Vec::with_capacity(self.config.num_hidden_layers * 2);
        for _ in 0..(self.config.num_hidden_layers * 2) {
            let shape = vec![1, self.config.num_key_value_heads, 0, self.config.head_dim()];
            let data: Vec<f32> = Vec::new();
            cache.push(Value::from_array((shape, data))
                .map_err(|e| anyhow::anyhow!("Failed to create empty cache: {:?}", e))?
                .into_dyn());
        }
        Ok(cache)
    }
}
