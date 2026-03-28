use crate::config::QwenConfig;
use anyhow::Result;
use ort::session::Session;
use ort::value::{DynValue, Value};
use std::path::Path;

use image::{DynamicImage, GenericImageView};
use ndarray::{Array2, Array4, s};

fn smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> (u32, u32) {
    let mut h = height;
    let mut w = width;
    if h < factor || w < factor {
        let scale = (factor as f32 / h as f32).max(factor as f32 / w as f32);
        h = (h as f32 * scale).round() as u32;
        w = (w as f32 * scale).round() as u32;
    }

    let mut h_bar = (h as f32 / factor as f32).round() as u32 * factor;
    let mut w_bar = (w as f32 / factor as f32).round() as u32 * factor;

    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f32 / max_pixels as f32).sqrt();
        h_bar = factor.max(((height as f32 / beta / factor as f32).floor() as u32) * factor);
        w_bar = factor.max(((width as f32 / beta / factor as f32).floor() as u32) * factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (height * width) as f32).sqrt();
        h_bar = ((height as f32 * beta / factor as f32).ceil() as u32) * factor;
        w_bar = ((width as f32 * beta / factor as f32).ceil() as u32) * factor;
    }
    (w_bar, h_bar)
}

fn preprocess_image(
    image: &DynamicImage,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(Array2<f32>, [i64; 3])> {
    let (width, height) = image.dimensions();
    let factor = (patch_size * merge_size) as u32;
    let (new_width, new_height) = smart_resize(height, width, factor, min_pixels, max_pixels);

    let resized = image.resize_exact(new_width, new_height, image::imageops::FilterType::Triangle);
    let mut pixels = Array4::<f32>::zeros((1, 3, new_height as usize, new_width as usize));

    for (x, y, color) in resized.pixels() {
        pixels[[0, 0, y as usize, x as usize]] = (color[0] as f32 / 255.0 - 0.5) / 0.5;
        pixels[[0, 1, y as usize, x as usize]] = (color[1] as f32 / 255.0 - 0.5) / 0.5;
        pixels[[0, 2, y as usize, x as usize]] = (color[2] as f32 / 255.0 - 0.5) / 0.5;
    }

    let mut patches = Array4::<f32>::zeros((temporal_patch_size, 3, new_height as usize, new_width as usize));
    for t in 0..temporal_patch_size {
        patches.slice_mut(s![t, .., .., ..]).assign(&pixels.slice(s![0, .., .., ..]));
    }

    let grid_t = patches.shape()[0] / temporal_patch_size;
    let grid_h = patches.shape()[2] / patch_size;
    let grid_w = patches.shape()[3] / patch_size;
    
    let channel = 3;
    let patch_dim = channel * temporal_patch_size * patch_size * patch_size;
    let num_patches = grid_t * grid_h * grid_w;
    let mut flattened = Array2::<f32>::zeros((num_patches, patch_dim));

    let mut idx = 0;
    for t in 0..grid_t {
        for h in 0..grid_h {
            for w in 0..grid_w {
                let mut p_idx = 0;
                for c in 0..channel {
                    for tp in 0..temporal_patch_size {
                        for ph in 0..patch_size {
                            for pw in 0..patch_size {
                                flattened[[idx, p_idx]] = patches[[t * temporal_patch_size + tp, c, h * patch_size + ph, w * patch_size + pw]];
                                p_idx += 1;
                            }
                        }
                    }
                }
                idx += 1;
            }
        }
    }

    Ok((flattened, [grid_t as i64, grid_h as i64, grid_w as i64]))
}

pub struct QwenEngine {
    pub embed_session: Session,
    pub decoder_session: Session,
    pub vision_session: Option<Session>,
    pub config: QwenConfig,
}

impl QwenEngine {
    pub fn new(model_dir: &Path, config: QwenConfig) -> Result<Self> {
        let onnx_dir = model_dir.join("onnx");
        
        let embed_session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{:?}", e))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?
            .commit_from_file(onnx_dir.join("embed_tokens_q4.onnx"))
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;

        let decoder_session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{:?}", e))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?
            .commit_from_file(onnx_dir.join("decoder_model_merged_q4.onnx"))
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;

        let vision_path = onnx_dir.join("vision_encoder_fp16.onnx");
        let vision_session = if vision_path.exists() {
            Some(Session::builder()
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
                .with_intra_threads(4)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
                .commit_from_file(vision_path)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?)
        } else {
            None
        };

        Ok(Self { embed_session, decoder_session, vision_session, config })
    }

    pub fn encode_image(&mut self, image: &DynamicImage) -> Result<(Vec<f32>, Vec<usize>, [i64; 3])> {
        let vision_session = self.vision_session.as_mut().ok_or_else(|| anyhow::anyhow!("Vision encoder not loaded"))?;
        let v_conf = self.config.vision_config.as_ref().ok_or_else(|| anyhow::anyhow!("Vision config missing"))?;
        
        let (pixel_values, grid_thw) = preprocess_image(
            image,
            v_conf.patch_size,
            v_conf.temporal_patch_size,
            v_conf.spatial_merge_size,
            56 * 56,
            16384 * 16384,
        )?;

        // pixel_values 应该是 [num_patches, patch_dim]
        let pixel_values_tensor = Value::from_array((pixel_values.shape().to_vec(), pixel_values.into_raw_vec())).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        // grid_thw 应该对齐 Transformers.js，可能需要 [1, 3] 形状
        let grid_thw_tensor = Value::from_array((vec![1, 3], grid_thw.to_vec())).map_err(|e| anyhow::anyhow!("{:?}", e))?;

        let outputs = vision_session.run(vec![
            ("pixel_values".to_string(), pixel_values_tensor.into_dyn()),
            ("image_grid_thw".to_string(), grid_thw_tensor.into_dyn()),
        ]).map_err(|e| anyhow::anyhow!("{:?}", e))?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        Ok((data.to_vec(), shape_usize, grid_thw))
    }

    pub fn forward_with_vision(
        &mut self,
        input_ids: Vec<i64>,
        image_embeds: Vec<f32>,
        image_embed_shape: Vec<usize>,
        image_token_indices: Vec<usize>,
        past_key_values: &mut Vec<DynValue>,
    ) -> Result<u32> {
        let batch_size = 1;
        let seq_len = input_ids.len();

        let input_ids_tensor = Value::from_array((vec![batch_size, seq_len], input_ids)).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let embed_outputs = self.embed_session.run(vec![
            ("input_ids".to_string(), input_ids_tensor.into_dyn())
        ]).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        
        let (text_shape, text_data) = embed_outputs[0].try_extract_tensor::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let hidden_dim = text_shape[2] as usize;
        
        let mut final_embeds = Vec::new();
        let img_start = image_token_indices[0];
        
        // 1. 图像前的文本向量
        final_embeds.extend_from_slice(&text_data[..img_start * hidden_dim]);
        // 2. 插入图像编码
        final_embeds.extend_from_slice(&image_embeds);
        // 3. 图像后的文本向量 (跳过 placeholder)
        final_embeds.extend_from_slice(&text_data[(img_start + 1) * hidden_dim..]);

        let final_seq_len = final_embeds.len() / hidden_dim;
        let inputs_embeds = Value::from_array((vec![batch_size, final_seq_len, hidden_dim], final_embeds)).map_err(|e| anyhow::anyhow!("{:?}", e))?;

        let mut all_inputs = std::collections::HashMap::new();
        all_inputs.insert("inputs_embeds".to_string(), inputs_embeds.into_dyn());
        all_inputs.insert("attention_mask".to_string(), Value::from_array((vec![batch_size, final_seq_len], vec![1i64; final_seq_len])).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());
        
        let mut pos_ids = Vec::with_capacity(final_seq_len);
        for i in 0..final_seq_len { pos_ids.push(i as i64); }
        let mut pos_ids_3d = Vec::with_capacity(3 * final_seq_len);
        for _ in 0..3 { pos_ids_3d.extend_from_slice(&pos_ids); }
        all_inputs.insert("position_ids".to_string(), Value::from_array((vec![3, batch_size, final_seq_len], pos_ids_3d)).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());
        
        all_inputs.insert("use_cache_branch".to_string(), Value::from_array((vec![1], vec![false])).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());

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

        let mut inputs_map = Vec::new();
        for input in self.decoder_session.inputs() {
            let name = input.name();
            if let Some(val) = all_inputs.remove(name) {
                inputs_map.push((name.to_string(), val));
            }
        }

        let mut outputs = self.decoder_session.run(inputs_map).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let logits_value = outputs.remove("logits").ok_or_else(|| anyhow::anyhow!("Missing logits"))?;
        let (_, logits_data) = logits_value.try_extract_tensor::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let vocab_size = self.config.vocab_size();
        let last_token_logits = &logits_data[logits_data.len() - vocab_size..];
        let next_token = last_token_logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32).unwrap();

        for i in 0..self.config.num_hidden_layers() {
            let layer_type = self.config.layer_type(i);
            if layer_type == "full_attention" {
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

    pub fn forward(
        &mut self,
        input_ids: Vec<i64>,
        past_key_values: &mut Vec<DynValue>,
        position_ids: Vec<i64>,
        is_prefill: bool,
    ) -> Result<u32> {
        let seq_len = input_ids.len();
        let batch_size = 1;

        let input_ids_tensor = Value::from_array((vec![batch_size, seq_len], input_ids)).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let embed_outputs = self.embed_session.run(vec![
            ("input_ids".to_string(), input_ids_tensor.into_dyn())
        ]).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        
        let (embed_shape, embed_data) = embed_outputs[0].try_extract_tensor::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let inputs_embeds = Value::from_array((embed_shape.to_vec(), embed_data.to_vec())).map_err(|e| anyhow::anyhow!("{:?}", e))?;

        let mask_len = if is_prefill { seq_len } else { 
            position_ids[0] as usize + 1
        };
        
        let mut all_inputs = std::collections::HashMap::new();
        all_inputs.insert("inputs_embeds".to_string(), inputs_embeds.into_dyn());
        all_inputs.insert("attention_mask".to_string(), Value::from_array((vec![batch_size, mask_len], vec![1i64; mask_len])).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());
        
        let mut pos_ids_3d = Vec::with_capacity(3 * seq_len);
        for _ in 0..3 { pos_ids_3d.extend_from_slice(&position_ids); }
        all_inputs.insert("position_ids".to_string(), Value::from_array((vec![3, batch_size, seq_len], pos_ids_3d)).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());
        
        all_inputs.insert("use_cache_branch".to_string(), Value::from_array((vec![1], vec![!is_prefill])).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());

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

        let mut inputs_map = Vec::new();
        for input in self.decoder_session.inputs() {
            let name = input.name();
            if let Some(val) = all_inputs.remove(name) {
                inputs_map.push((name.to_string(), val));
            }
        }

        let mut outputs = self.decoder_session.run(inputs_map).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let logits_value = outputs.remove("logits").ok_or_else(|| anyhow::anyhow!("Missing logits"))?;
        let (_, logits_data) = logits_value.try_extract_tensor::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?;
        
        let vocab_size = self.config.vocab_size();
        let last_token_logits = &logits_data[logits_data.len() - vocab_size..];
        
        let next_token = last_token_logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32).unwrap();

        for i in 0..self.config.num_hidden_layers() {
            let layer_type = self.config.layer_type(i);
            if layer_type == "full_attention" {
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
                cache.push(Value::from_array((vec![1, heads, 1, dim], vec![0.0f32; 1 * heads * 1 * dim])).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());
                cache.push(Value::from_array((vec![1, heads, 1, dim], vec![0.0f32; 1 * heads * 1 * dim])).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());
            } else {
                let key_dim = self.config.text_config.linear_num_key_heads * self.config.text_config.linear_key_head_dim;
                let value_dim = self.config.text_config.linear_num_value_heads * self.config.text_config.linear_value_head_dim;
                let conv_dim = key_dim * 2 + value_dim;
                let kernel_dim = self.config.text_config.linear_conv_kernel_dim;
                let heads = self.config.text_config.linear_num_value_heads;
                let dim_k = self.config.text_config.linear_key_head_dim;
                let dim_v = self.config.text_config.linear_value_head_dim;

                cache.push(Value::from_array((vec![1, conv_dim, kernel_dim], vec![0.0f32; 1 * conv_dim * kernel_dim])).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());
                cache.push(Value::from_array((vec![1, heads, dim_k, dim_v], vec![0.0f32; 1 * heads * dim_k * dim_v])).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());
            }
        }
        Ok(cache)
    }
}
