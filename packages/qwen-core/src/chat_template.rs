use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// OpenAI-compatible message content item
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    #[serde(rename = "image")]
    Image {},
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Message content can be a plain string or an array of content parts
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// Tool call function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Tool call in assistant message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

/// Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// A chat message (OpenAI-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// For tool role messages
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Create a simple user text message
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(MessageContent::Text(text.into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a user message with image + text (multimodal)
    pub fn user_with_image(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Image {},
                ContentPart::Text { text: text.into() },
            ])),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a system message
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(MessageContent::Text(text.into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(MessageContent::Text(text.into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// Parameters for rendering a chat template
pub struct ChatTemplateParams<'a> {
    pub messages: &'a [ChatMessage],
    pub tools: Option<&'a [Tool]>,
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
}

/// Chat template engine using minijinja
pub struct ChatTemplateEngine {
    template_source: String,
}

impl ChatTemplateEngine {
    /// Load chat_template from a .jinja file in the model directory
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        // Try loading from file
        let template_path = model_dir.join("chat_template.jinja");
        if template_path.exists() {
            let source = std::fs::read_to_string(&template_path)
                .with_context(|| format!("Failed to read chat template: {:?}", template_path))?;
            return Ok(Self { template_source: source });
        }

        // Try extracting from tokenizer_config.json
        let config_path = model_dir.join("tokenizer_config.json");
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(template) = config.get("chat_template").and_then(|v| v.as_str()) {
                return Ok(Self { template_source: template.to_string() });
            }
        }

        // Fallback: use the bundled template
        let bundled = include_str!("../chat_template.jinja");
        Ok(Self { template_source: bundled.to_string() })
    }

    /// Create from a template string directly
    pub fn from_string(template: String) -> Self {
        Self { template_source: template }
    }

    /// Render messages into a formatted prompt string
    pub fn render(&self, params: &ChatTemplateParams) -> Result<String> {
        let mut env = minijinja::Environment::new();

        // Support string methods (startswith, endswith, etc.) used in the Jinja template
        env.set_unknown_method_callback(
            |_state: &minijinja::State, value: &minijinja::Value, method: &str, args: &[minijinja::Value]| -> Result<minijinja::Value, minijinja::Error> {
                if let Some(s) = value.as_str() {
                    if method == "startswith" {
                        if let Some(prefix) = args.get(0).and_then(|v| v.as_str()) {
                            return Ok(minijinja::Value::from(s.starts_with(prefix)));
                        }
                    } else if method == "endswith" {
                        if let Some(suffix) = args.get(0).and_then(|v| v.as_str()) {
                            return Ok(minijinja::Value::from(s.ends_with(suffix)));
                        }
                    } else if method == "split" {
                        if let Some(sep) = args.get(0).and_then(|v| v.as_str()) {
                            let parts: Vec<minijinja::Value> = s.split(sep)
                                .map(|p| minijinja::Value::from(p))
                                .collect();
                            return Ok(minijinja::Value::from(parts));
                        }
                    } else if method == "rstrip" {
                        if let Some(chars) = args.get(0).and_then(|v| v.as_str()) {
                            let mut result = s;
                            for c in chars.chars() {
                                result = result.trim_end_matches(c);
                            }
                            return Ok(minijinja::Value::from(result));
                        } else {
                            return Ok(minijinja::Value::from(s.trim_end()));
                        }
                    } else if method == "lstrip" {
                        if let Some(chars) = args.get(0).and_then(|v| v.as_str()) {
                            let mut result = s;
                            for c in chars.chars() {
                                result = result.trim_start_matches(c);
                            }
                            return Ok(minijinja::Value::from(result));
                        } else {
                            return Ok(minijinja::Value::from(s.trim_start()));
                        }
                    }
                }
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::UnknownMethod,
                    format!("unknown method: string has no method named {}", method),
                ))
            },
        );

        // Register items filter for iterating over object key-value pairs
        env.add_filter(
            "items",
            |value: minijinja::Value| -> Result<Vec<minijinja::Value>, minijinja::Error> {
                if value.kind() == minijinja::value::ValueKind::Map {
                    let mut items = Vec::new();
                    if let Ok(iter) = value.try_iter() {
                        for key in iter {
                            let val = value.get_attr(key.as_str().unwrap_or_default())
                                .unwrap_or(minijinja::Value::UNDEFINED);
                            items.push(minijinja::Value::from(vec![key, val]));
                        }
                    }
                    Ok(items)
                } else {
                    Err(minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "items filter requires a mapping",
                    ))
                }
            },
        );

        // Register raise_exception function
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );

        env.add_template("chat_template", &self.template_source)
            .context("Failed to compile chat template")?;

        let tmpl = env.get_template("chat_template")
            .context("Failed to get chat template")?;

        // Convert messages to minijinja-friendly values
        let messages_val = serde_json::to_value(params.messages)?;
        let messages_jinja = minijinja::Value::from_serialize(&messages_val);

        let tools_jinja = if let Some(tools) = params.tools {
            let tools_val = serde_json::to_value(tools)?;
            minijinja::Value::from_serialize(&tools_val)
        } else {
            minijinja::Value::from(false)
        };

        let rendered = tmpl.render(minijinja::context! {
            messages => messages_jinja,
            tools => tools_jinja,
            add_generation_prompt => params.add_generation_prompt,
            enable_thinking => params.enable_thinking,
            add_vision_id => true,
        }).context("Failed to render chat template")?;

        Ok(rendered)
    }

    /// Count how many image placeholders are in the rendered prompt
    pub fn count_images(rendered: &str) -> usize {
        rendered.matches("<|image_pad|>").count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_text_message() {
        let engine = ChatTemplateEngine::from_string(include_str!("../chat_template.jinja").to_string());
        let messages = vec![ChatMessage::user("Hello!")];
        let result = engine.render(&ChatTemplateParams {
            messages: &messages,
            tools: None,
            add_generation_prompt: true,
            enable_thinking: false,
        }).unwrap();
        assert!(result.contains("Hello!"));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_image_message() {
        let engine = ChatTemplateEngine::from_string(include_str!("../chat_template.jinja").to_string());
        let messages = vec![ChatMessage::user_with_image("Describe this image")];
        let result = engine.render(&ChatTemplateParams {
            messages: &messages,
            tools: None,
            add_generation_prompt: true,
            enable_thinking: false,
        }).unwrap();
        assert!(result.contains("<|image_pad|>"));
        assert!(result.contains("Describe this image"));
    }

    #[test]
    fn test_thinking_enabled() {
        let engine = ChatTemplateEngine::from_string(include_str!("../chat_template.jinja").to_string());
        let messages = vec![ChatMessage::user("Think about this")];
        let result = engine.render(&ChatTemplateParams {
            messages: &messages,
            tools: None,
            add_generation_prompt: true,
            enable_thinking: true,
        }).unwrap();
        // When enable_thinking is true, it should end with <think>\n
        assert!(result.contains("<think>\n"));
        // Should not have the immediate close tag
        assert!(!result.contains("<think>\n\n</think>"));
    }
}
