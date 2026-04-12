use crate::chat_template::{ChatMessage, ChatTemplateEngine, ChatTemplateParams, Tool};
use crate::config::QwenConfig;
use crate::model::QwenEngine;
use anyhow::Result;
use image::DynamicImage;
use ort::value::DynValue;
use std::path::Path;
use tokenizers::Tokenizer;

/// Generation parameters
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    pub enable_thinking: bool,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            enable_thinking: false,
        }
    }
}

/// Result of a streaming generation callback
pub enum StreamAction {
    Continue,
    Stop,
}

/// High-level inference pipeline wrapping Engine + Tokenizer + ChatTemplate
pub struct InferencePipeline {
    pub engine: QwenEngine,
    pub tokenizer: Tokenizer,
    pub template_engine: ChatTemplateEngine,
    pub config: QwenConfig,
}

impl InferencePipeline {
    /// Load all components from a model directory
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = QwenConfig::from_file(model_dir.join("config.json"))?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let engine = QwenEngine::new(model_dir, config.clone())?;
        let template_engine = ChatTemplateEngine::from_model_dir(model_dir)?;

        Ok(Self {
            engine,
            tokenizer,
            template_engine,
            config,
        })
    }

    /// Apply chat template and tokenize
    pub fn prepare_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        enable_thinking: bool,
    ) -> Result<(Vec<i64>, String)> {
        let rendered = self.template_engine.render(&ChatTemplateParams {
            messages,
            tools,
            add_generation_prompt: true,
            enable_thinking,
        })?;

        let image_pad_token_id = self.tokenizer.token_to_id("<|image_pad|>")
            .map(|id| id as i64)
            .unwrap_or_else(|| self.config.image_token_id.unwrap_or(151655) as i64);

        let mut input_ids = Vec::new();
        let parts: Vec<&str> = rendered.split("<|image_pad|>").collect();
        for (i, part) in parts.iter().enumerate() {
            if !part.is_empty() {
                let encoding = self.tokenizer.encode(*part, false)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                input_ids.extend(encoding.get_ids().iter().map(|&x| x as i64));
            }
            if i < parts.len() - 1 {
                input_ids.push(image_pad_token_id);
            }
        }

        Ok((input_ids, rendered))
    }

    /// Generate a complete response, calling `on_token` for each generated token.
    /// `on_token` receives the decoded text for each token and returns a StreamAction.
    /// If an image is provided, it will be encoded and used for vision inference.
    pub fn generate<F>(
        &mut self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        image: Option<&DynamicImage>,
        params: &GenerateParams,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str) -> StreamAction,
    {
        let (input_ids, _rendered) = self.prepare_prompt(messages, tools, params.enable_thinking)?;
        let mut kv_cache: Vec<DynValue> = self.engine.create_empty_kv_cache()?;

        let eos_id = self.config.get_eos_id();
        let image_pad_token_id = self.tokenizer.token_to_id("<|image_pad|>")
            .map(|id| id as i64)
            .unwrap_or_else(|| self.config.image_token_id.unwrap_or(151655) as i64);

        // Find image token indices
        let image_token_indices: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == image_pad_token_id)
            .map(|(i, _)| i)
            .collect();

        let mut next_token: u32;
        let mut current_pos: i64;
        let mut kv_len: usize;

        // Prefill
        if let (Some(img), true) = (image, !image_token_indices.is_empty()) {
            let (image_embeds, _shape, thw) = self.engine.encode_image(img)?;
            let n_image_tokens = image_embeds.len() / self.config.text_config.hidden_size;
            let result = self.engine.forward_with_vision(
                input_ids.clone(),
                image_embeds,
                thw,
                image_token_indices,
                &mut kv_cache,
            )?;
            next_token = result.0;
            current_pos = result.1 + 1;
            kv_len = input_ids.len() - 1 + n_image_tokens;
        } else {
            let seq_len = input_ids.len();
            next_token = self.engine.forward(
                input_ids,
                &mut kv_cache,
                (0..seq_len as i64).collect(),
                seq_len,
                true,
            )?;
            current_pos = seq_len as i64;
            kv_len = seq_len;
        }

        // Decode loop
        let mut full_output = String::new();
        for _ in 0..params.max_tokens {
            if next_token == eos_id {
                break;
            }

            let token_text = self.tokenizer.decode(&[next_token], true)
                .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

            full_output.push_str(&token_text);

            match on_token(&token_text) {
                StreamAction::Continue => {},
                StreamAction::Stop => break,
            }

            next_token = self.engine.forward(
                vec![next_token as i64],
                &mut kv_cache,
                vec![current_pos],
                kv_len + 1,
                false,
            )?;
            current_pos += 1;
            kv_len += 1;
        }

        Ok(full_output)
    }
}
