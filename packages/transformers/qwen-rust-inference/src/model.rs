use crate::config::QwenConfig;
use anyhow::Result;
use ort::session::Session;
use ort::value::{DynValue, Value};
use std::path::Path;

pub struct QwenEngine {
    pub embed_session: Session,
    pub decoder_session: Session,
    pub config: QwenConfig,
}

impl QwenEngine {
    pub fn new(model_dir: &Path, config: QwenConfig) -> Result<Self> {
        let onnx_dir = model_dir.join("onnx");
        
        let embed_session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create builder: {:?}", e))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("Failed to set threads: {:?}", e))?
            .commit_from_file(onnx_dir.join("embed_tokens_q4.onnx"))
            .map_err(|e| anyhow::anyhow!("Failed to load embed_tokens: {:?}", e))?;

        let decoder_session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create builder: {:?}", e))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("Failed to set threads: {:?}", e))?
            .commit_from_file(onnx_dir.join("decoder_model_merged_q4.onnx"))
            .map_err(|e| anyhow::anyhow!("Failed to load decoder: {:?}", e))?;

        Ok(Self { embed_session, decoder_session, config })
    }

    pub fn forward(
        &mut self,
        input_ids: Vec<i64>,
        past_key_values: &mut Vec<DynValue>,
        position_ids: Vec<i64>,
        is_prefill: bool,
    ) -> Result<u32> {
        let seq_len = input_ids.len();
        let batch_size = 1;

        let input_ids_tensor = Value::from_array((vec![batch_size, seq_len], input_ids))
            .map_err(|e| anyhow::anyhow!("Failed to create input_ids tensor: {:?}", e))?;
        
        let embed_outputs = self.embed_session.run(vec![
            ("input_ids".to_string(), input_ids_tensor.into_dyn())
        ]).map_err(|e| anyhow::anyhow!("Embedding run failed: {:?}", e))?;
        
        let (embed_shape, embed_data) = embed_outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract embeds: {:?}", e))?;
        let inputs_embeds = Value::from_array((embed_shape.to_vec(), embed_data.to_vec()))
            .map_err(|e| anyhow::anyhow!("Failed to create inputs_embeds: {:?}", e))?;

        // Calculate attention mask length based on actual KV cache contents
        // This is simplified for MVP; in mixed models, the logic depends on whether we use_cache_branch.
        let kv_len = if is_prefill { 0 } else { 
            // Simplified: in Qwen3.5, attention_mask usually covers the cumulative seq_len
            // We'll track it in the main loop instead of calculating from past_key_values here
            0 
        };
        // Let's take the mask_len as a parameter or calculate correctly from position_ids for now
        let mask_len = if is_prefill { seq_len } else { 
            // We will fix this in main.rs to pass the correct cumulative length
            position_ids[0] as usize + 1
        };
        
        let attention_mask = Value::from_array((vec![batch_size, mask_len], vec![1i64; mask_len]))?;
        let position_tensor = Value::from_array((vec![batch_size, seq_len], position_ids))?;
        let use_cache_branch = Value::from_array((vec![1], vec![!is_prefill]))?;

        let mut inputs_map = vec![
            ("inputs_embeds".to_string(), inputs_embeds.into_dyn()),
            ("attention_mask".to_string(), attention_mask.into_dyn()),
            ("position_ids".to_string(), position_tensor.into_dyn()),
            ("use_cache_branch".to_string(), use_cache_branch.into_dyn()),
        ];

        for i in 0..self.config.num_hidden_layers() {
            let k = past_key_values.remove(0);
            let v = past_key_values.remove(0);
            inputs_map.push((format!("past_key_values.{}.key", i), k));
            inputs_map.push((format!("past_key_values.{}.value", i), v));
        }

        let mut outputs = self.decoder_session.run(inputs_map)
            .map_err(|e| anyhow::anyhow!("Decoder run failed: {:?}", e))?;

        let logits_value = outputs.remove("logits")
            .ok_or_else(|| anyhow::anyhow!("Missing logits"))?;
        let (_, logits_data) = logits_value.try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract logits: {:?}", e))?;
        
        let vocab_size = self.config.text_config.vocab_size;
        let last_token_logits = &logits_data[logits_data.len() - vocab_size..];
        
        let next_token = last_token_logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32).unwrap();

        for i in 0..self.config.num_hidden_layers() {
            let pk = outputs.remove(format!("present.{}.key", i))
                .ok_or_else(|| anyhow::anyhow!("Missing present key layer {}", i))?;
            let pv = outputs.remove(format!("present.{}.value", i))
                .ok_or_else(|| anyhow::anyhow!("Missing present value layer {}", i))?;
            past_key_values.push(pk);
            past_key_values.push(pv);
        }

        Ok(next_token)
    }

    pub fn create_empty_kv_cache(&self) -> Result<Vec<DynValue>> {
        let n_layers = self.config.num_hidden_layers();
        let mut cache = Vec::with_capacity(n_layers * 2);
        for i in 0..n_layers {
            let layer_type = self.config.layer_type(i);
            let (shape_k, shape_v) = if layer_type == "linear_attention" {
                // Linear attention layers in Qwen3.5 usually have a fixed state size (e.g. 1, 16, 128, 128)
                // We use 1 instead of 0 for initial dummy dimensions if needed, 
                // but let's try to find the exact required shape from config.
                let heads = self.config.text_config.linear_num_key_heads;
                let dim_k = self.config.text_config.linear_key_head_dim;
                let dim_v = self.config.text_config.linear_key_head_dim; // or value_head_dim
                // Linear attention states are often [batch, heads, d_model/heads, head_dim] 
                // but for ONNX export, check optimum's behavior.
                // Usually it's [1, heads, 128, 128] or similar.
                // Based on "dimension #3" error, if we use 0 it might fail.
                (vec![1, heads, 128, 128], vec![1, heads, 128, 128])
            } else {
                // Standard full attention
                let heads = self.config.text_config.num_key_value_heads;
                let dim = self.config.text_config.head_dim;
                // If 0 is not allowed, we use 1 with a mask, but many ONNX models 
                // require EXACTLY what the graph defines. 
                // If the error says dimension #3, let's ensure head_dim is > 0.
                (vec![1, heads, 0, dim], vec![1, heads, 0, dim])
            };

            // If ort strictly forbids 0-sized tensors in from_array:
            let k_tensor = if shape_k[2] == 0 {
                // Use a trick to create a 0-sized tensor if from_array fails
                Value::from_array((vec![1, shape_k[1], 0, shape_k[3]], Vec::<f32>::new()))?
            } else {
                Value::from_array((shape_k, vec![0f32; 1 * 16 * 128 * 128]))? // Placeholder size
            };
            
            let v_tensor = if shape_v[2] == 0 {
                Value::from_array((vec![1, shape_v[1], 0, shape_v[3]], Vec::<f32>::new()))?
            } else {
                Value::from_array((shape_v, vec![0f32; 1 * 16 * 128 * 128]))?
            };

            cache.push(k_tensor.into_dyn());
            cache.push(v_tensor.into_dyn());
        }
        Ok(cache)
    }
}
