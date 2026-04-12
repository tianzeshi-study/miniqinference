use crate::config::QwenConfig;
use anyhow::Result;
use ort::session::Session;
use ort::value::{DynValue, Value};
use std::path::Path;

use image::{DynamicImage, GenericImageView};
use ndarray::{Array2, Array4};

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

    // 1. 高质量缩放
    let resized = image.resize_exact(new_width, new_height, image::imageops::FilterType::CatmullRom);
    
    // 2. 转换为 [T, C, H, W] 张量
    // Qwen2-VL 预处理会将一张图在时间维度上复制或拆分，通常静态图 T = temporal_patch_size
    let mut patches = Array4::<f32>::zeros((temporal_patch_size, 3, new_height as usize, new_width as usize));
    let mean = [0.5, 0.5, 0.5];
    let std = [0.5, 0.5, 0.5];
    
    for (x, y, color) in resized.pixels() {
        for t in 0..temporal_patch_size {
            for c in 0..3 {
                patches[[t, c, y as usize, x as usize]] = (color[c] as f32 / 255.0 - mean[c]) / std[c];
            }
        }
    }

    // 3. 计算维度
    let channel = 3;
    let grid_t = 1; // 经过时间轴合并后，逻辑上的 T 变为 1
    let grid_h = new_height as usize / patch_size;
    let grid_w = new_width as usize / patch_size;
    let gh_m = grid_h / merge_size;
    let gw_m = grid_w / merge_size;

    // 4. 执行重排与展平 (等效于 JS 的 .permute(0, 3, 6, 4, 7, 2, 1, 5, 8))
    let mut flattened = Array2::<f32>::zeros((
        grid_t * grid_h * grid_w, 
        channel * temporal_patch_size * patch_size * patch_size
    ));

    let mut idx = 0;
    // 严格按照 Transformers.js 的空间块排列顺序
    for i in 0..gh_m {
        for j in 0..gw_m {
            for m_h in 0..merge_size {
                for m_w in 0..merge_size {
                    let mut p_idx = 0;
                    // 核心特征向量构造：C -> T -> PH -> PW
                    for c in 0..channel {
                        for tp in 0..temporal_patch_size {
                            for ph in 0..patch_size {
                                for pw in 0..patch_size {
                                    let h = (i * merge_size + m_h) * patch_size + ph;
                                    let w = (j * merge_size + m_w) * patch_size + pw;
                                    flattened[[idx, p_idx]] = patches[[tp, c, h, w]];
                                    p_idx += 1;
                                }
                            }
                        }
                    }
                    idx += 1;
                }
            }
        }
    }

    // 返回 grid_thw，注意 Qwen2-VL 期望的是合并后的 grid 尺寸
    // 即 [1, grid_h, grid_w]
    
    println!("=== RUST PREPROCESSING DEBUG ===");
    println!("flatten_patches shape: {:?}", flattened.shape());
    
    if let Some(slice) = flattened.as_slice() {
        let len = slice.len();
        let first_10: Vec<f32> = slice.iter().take(10).cloned().collect();
        let last_10: Vec<f32> = if len > 10 {
            slice[len - 10..].to_vec()
        } else {
            slice.to_vec()
        };
        println!("flatten_patches first 10: {:?}", first_10);
        println!("flatten_patches last 10: {:?}", last_10);
    }
    
    let sum: f32 = flattened.iter().sum();
    println!("flatten_patches sum: {}", sum);
    
    println!("image_grid_thw data: {:?}", [1i64, grid_h as i64, grid_w as i64]);
    println!("==============================\n");

    Ok((flattened, [1i64, grid_h as i64, grid_w as i64]))
}
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
            256 * 256,
            4096 * 4096,
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
        grid_thw: [i64; 3],
        image_token_indices: Vec<usize>,
        past_key_values: &mut Vec<DynValue>,
    ) -> Result<(u32, i64)> {
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
        
        let mut t_ids = Vec::with_capacity(final_seq_len);
        let mut h_ids = Vec::with_capacity(final_seq_len);
        let mut w_ids = Vec::with_capacity(final_seq_len);

        for i in 0..img_start {
            t_ids.push(i as i64);
            h_ids.push(i as i64);
            w_ids.push(i as i64);
        }

        let spatial_merge_size = self.config.vision_config.as_ref().map(|v| v.spatial_merge_size).unwrap_or(2) as i64;
        let llm_grid_t = grid_thw[0];
        let llm_grid_h = grid_thw[1] / spatial_merge_size;
        let llm_grid_w = grid_thw[2] / spatial_merge_size;
        let grid_size = (llm_grid_t * llm_grid_h * llm_grid_w) as usize;
        let offset = img_start as i64;

        for i in 0..grid_size as i64 {
            t_ids.push(offset + i / (llm_grid_h * llm_grid_w));
            h_ids.push(offset + (i / llm_grid_w) % llm_grid_h);
            w_ids.push(offset + i % llm_grid_w);
        }

        let max_img_pos = llm_grid_t.max(llm_grid_h).max(llm_grid_w);
        let st_idx = offset + max_img_pos;
        let remaining = final_seq_len.saturating_sub(img_start + grid_size);
        for i in 0..remaining as i64 {
            t_ids.push(st_idx + i);
            h_ids.push(st_idx + i);
            w_ids.push(st_idx + i);
        }

        let max_pos = if remaining > 0 {
            st_idx + remaining as i64 - 1
        } else if grid_size > 0 {
            offset + max_img_pos - 1
        } else {
            img_start as i64 - 1
        };

        let mut pos_ids_3d = Vec::with_capacity(3 * final_seq_len);
        pos_ids_3d.extend_from_slice(&t_ids);
        pos_ids_3d.extend_from_slice(&h_ids);
        pos_ids_3d.extend_from_slice(&w_ids);
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

        Ok((next_token, max_pos))
    }

    pub fn forward(
        &mut self,
        input_ids: Vec<i64>,
        past_key_values: &mut Vec<DynValue>,
        position_ids: Vec<i64>,
        mask_len: usize,
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
                cache.push(Value::from_array((vec![1, heads, 1, dim], vec![0.0f32; heads * dim])).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());
                cache.push(Value::from_array((vec![1, heads, 1, dim], vec![0.0f32; heads * dim])).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());
            } else {
                let key_dim = self.config.text_config.linear_num_key_heads * self.config.text_config.linear_key_head_dim;
                let value_dim = self.config.text_config.linear_num_value_heads * self.config.text_config.linear_value_head_dim;
                let conv_dim = key_dim * 2 + value_dim;
                let kernel_dim = self.config.text_config.linear_conv_kernel_dim;
                let heads = self.config.text_config.linear_num_value_heads;
                let dim_k = self.config.text_config.linear_key_head_dim;
                let dim_v = self.config.text_config.linear_value_head_dim;

                cache.push(Value::from_array((vec![1, conv_dim, kernel_dim], vec![0.0f32; conv_dim * kernel_dim])).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());
                cache.push(Value::from_array((vec![1, heads, dim_k, dim_v], vec![0.0f32; heads * dim_k * dim_v])).map_err(|e| anyhow::anyhow!("{:?}", e))?.into_dyn());
            }
        }
        Ok(cache)
    }
}
