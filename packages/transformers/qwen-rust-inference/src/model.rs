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

        // 1. Embedding
        let input_ids_tensor = Value::from_array((vec![batch_size, seq_len], input_ids))?;
        let embed_outputs = self.embed_session.run(vec![
            ("input_ids".to_string(), input_ids_tensor.into_dyn())
        ]).map_err(|e| anyhow::anyhow!("Embedding run failed: {:?}", e))?;
        
        let (embed_shape, embed_data) = embed_outputs[0].try_extract_tensor::<f32>()?;
        let inputs_embeds = Value::from_array((embed_shape.to_vec(), embed_data.to_vec()))?;

        // 2. 准备所有可能的输入
        let mask_len = if is_prefill { seq_len } else { 
            position_ids[0] as usize + 1
        };
        
        let mut all_inputs = std::collections::HashMap::new();
        
        // 基础输入
        all_inputs.insert("inputs_embeds".to_string(), inputs_embeds.into_dyn());
        all_inputs.insert("attention_mask".to_string(), Value::from_array((vec![batch_size, mask_len], vec![1i64; mask_len]))?.into_dyn());
        
        // 3D Position IDs (针对 Qwen 3.5)
        let mut pos_ids_3d = Vec::with_capacity(3 * seq_len);
        for _ in 0..3 { pos_ids_3d.extend_from_slice(&position_ids); }
        all_inputs.insert("position_ids".to_string(), Value::from_array((vec![3, batch_size, seq_len], pos_ids_3d))?.into_dyn());
        
        // 动态分支开关
        all_inputs.insert("use_cache_branch".to_string(), Value::from_array((vec![1], vec![!is_prefill]))?.into_dyn());

        // KV-Cache 输入
        for i in 0..self.config.num_hidden_layers() {
            let layer_type = self.config.layer_type(i);
            if layer_type == "full_attention" {
                all_inputs.insert(format!("past_key_values.{}.key", i), past_key_values.remove(0));
                all_inputs.insert(format!("past_key_values.{}.value", i), past_key_values.remove(0));
            } else if layer_type == "linear_attention" {
                all_inputs.insert(format!("past_conv.{}", i), past_key_values.remove(0));
                all_inputs.insert(format!("past_recurrent.{}", i), past_key_values.remove(0));
            }
        }

        // --- 核心修复：只保留模型声明过的输入 ---
        let mut inputs_map = Vec::new();
        for input in self.decoder_session.inputs() {
            let name = input.name();
            if let Some(val) = all_inputs.remove(name) {
                inputs_map.push((name.to_string(), val));
            }
        }

        let mut outputs = self.decoder_session.run(inputs_map)
            .map_err(|e| anyhow::anyhow!("Decoder run failed: {:?}", e))?;

        // 3. 提取 Logits 并采样
        let logits_value = outputs.remove("logits")
            .ok_or_else(|| anyhow::anyhow!("Missing logits"))?;
        let (_, logits_data) = logits_value.try_extract_tensor::<f32>()?;
        
        let vocab_size = self.config.vocab_size();
        let last_token_logits = &logits_data[logits_data.len() - vocab_size..];
        
        let next_token = last_token_logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32).unwrap();

        // 4. 更新 KV-cache
        for i in 0..self.config.num_hidden_layers() {
            let layer_type = self.config.layer_type(i);
            if layer_type == "full_attention" {
                // 有些模型输出名可能叫 present.i.key 或 present_key_values.i.key
                let k_name = format!("present.{}.key", i);
                let v_name = format!("present.{}.value", i);
                past_key_values.push(outputs.remove(k_name).or_else(|| outputs.remove(format!("present_key_values.{}.key", i))).unwrap());
                past_key_values.push(outputs.remove(v_name).or_else(|| outputs.remove(format!("present_key_values.{}.value", i))).unwrap());
            } else if layer_type == "linear_attention" {
                past_key_values.push(outputs.remove(format!("present_conv.{}", i)).unwrap());
                past_key_values.push(outputs.remove(format!("present_recurrent.{}", i)).unwrap());
            }
        }

        Ok(next_token)
    }

    pub fn create_empty_kv_cache(&self) -> Result<Vec<DynValue>> {
        let n_layers = self.config.num_hidden_layers();
        let mut cache = Vec::with_capacity(n_layers * 2);
        for i in 0..n_layers {
            let layer_type = self.config.layer_type(i);
            if layer_type == "full_attention" {
                let heads = self.config.text_config.num_key_value_heads;
                let dim = self.config.head_dim();
                // 既然 0 维度报错，我们使用 1 维度占位。Merged 模型在 prefill 分支下会自动处理它。
                cache.push(Value::from_array((vec![1, heads, 1, dim], vec![0.0f32; 1 * heads * 1 * dim]))?.into_dyn());
                cache.push(Value::from_array((vec![1, heads, 1, dim], vec![0.0f32; 1 * heads * 1 * dim]))?.into_dyn());
            } else {
                let key_dim = self.config.text_config.linear_num_key_heads * self.config.text_config.linear_key_head_dim;
                let value_dim = self.config.text_config.linear_num_value_heads * self.config.text_config.linear_value_head_dim;
                let conv_dim = key_dim * 2 + value_dim;
                let kernel_dim = self.config.text_config.linear_conv_kernel_dim;
                let heads = self.config.text_config.linear_num_value_heads;
                let dim_k = self.config.text_config.linear_key_head_dim;
                let dim_v = self.config.text_config.linear_value_head_dim;

                cache.push(Value::from_array((vec![1, conv_dim, kernel_dim], vec![0.0f32; 1 * conv_dim * kernel_dim]))?.into_dyn());
                cache.push(Value::from_array((vec![1, heads, dim_k, dim_v], vec![0.0f32; 1 * heads * dim_k * dim_v]))?.into_dyn());
            }
        }
        Ok(cache)
    }
}
