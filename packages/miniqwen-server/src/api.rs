use std::sync::Arc;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json, Response,
    },
};

use tokio::sync::Mutex;
use qwen_core::{
    ContentPart, GenerateParams, InferencePipeline,
    MessageContent, StreamAction,
};

use crate::types::*;

pub struct AppState {
    pub pipeline: Mutex<InferencePipeline>,
    pub model_name: String,
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let model_name = state.model_name.clone();

    let enable_thinking = req.enable_thinking.unwrap_or(false);
    let max_tokens = req.max_tokens.unwrap_or(2048);

    // Extract image from messages if present (find the first image_url in user messages)
    let image = extract_image_from_messages(&req.messages).await;

    let params = GenerateParams {
        max_tokens,
        enable_thinking,
    };

    let tools = req.tools.clone();
    let is_stream = req.stream;

    if is_stream {
        // Streaming mode
        handle_stream(state, req, request_id, created, model_name, image, params, tools).await
    } else {
        // Non-streaming mode
        handle_non_stream(state, req, request_id, created, model_name, image, params, tools).await
    }
}

/// Extract image from messages (supports base64 data URLs and local file paths)
async fn extract_image_from_messages(
    messages: &[qwen_core::ChatMessage],
) -> Option<image::DynamicImage> {
    for msg in messages {
        if msg.role != "user" {
            continue;
        }
        if let Some(MessageContent::Parts(parts)) = &msg.content {
            for part in parts {
                match part {
                    ContentPart::ImageUrl { image_url } => {
                        return load_image_from_url(&image_url.url).ok();
                    }
                    ContentPart::Image {} => {
                        // Image placeholder without URL — skip
                    }
                    _ => {}
                }
            }
        }
    }
    None
}

/// Load image from base64 data URL or file path
fn load_image_from_url(url: &str) -> anyhow::Result<image::DynamicImage> {
    if url.starts_with("data:image") {
        // base64 data URL: data:image/png;base64,xxxxx
        let parts: Vec<&str> = url.splitn(2, ",").collect();
        if parts.len() != 2 {
            anyhow::bail!("Invalid data URL format");
        }
        let decoded = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            parts[1],
        )?;
        let img = image::load_from_memory(&decoded)?;
        Ok(img)
    } else if url.starts_with("http://") || url.starts_with("https://") {
        anyhow::bail!("HTTP image URLs are not supported yet. Use base64 data URLs or local file paths.")
    } else {
        // Treat as local file path
        let img = image::open(url)?;
        Ok(img)
    }
}

/// Handle non-streaming response
async fn handle_non_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    request_id: String,
    created: i64,
    model_name: String,
    image: Option<image::DynamicImage>,
    params: GenerateParams,
    tools: Option<Vec<qwen_core::Tool>>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let mut pipeline = state.pipeline.lock().await;

    let result = tokio::task::block_in_place(|| {
        let tools_ref = tools.as_deref();
        let (input_ids, _) = pipeline
            .prepare_prompt(&req.messages, tools_ref, params.enable_thinking)
            .map_err(|e| e.to_string())?;
        let prompt_tokens = input_ids.len();

        let mut output = String::new();
        pipeline.generate(
            &req.messages,
            tools_ref,
            image.as_ref(),
            &params,
            |token| {
                output.push_str(token);
                StreamAction::Continue
            },
        ).map_err(|e| e.to_string())?;

        let completion_tokens = pipeline.tokenizer
            .encode(output.as_str(), false)
            .map(|e| e.get_ids().len())
            .unwrap_or(0);

        Ok::<_, String>((output, prompt_tokens, completion_tokens))
    });

    match result {
        Ok((output, prompt_tokens, completion_tokens)) => {
            // Parse thinking content if present
            let (reasoning, content) = parse_thinking_content(&output, params.enable_thinking);

            // Parse tool calls if present
            let tool_calls = parse_tool_calls(&content);
            let final_content = if tool_calls.is_some() {
                let clean = strip_tool_calls(&content);
                if clean.trim().is_empty() { None } else { Some(clean) }
            } else {
                Some(content)
            };

            let finish_reason = if tool_calls.is_some() {
                "tool_calls"
            } else {
                "stop"
            };

            let response = ChatCompletionResponse {
                id: request_id,
                object: "chat.completion".to_string(),
                created,
                model: model_name,
                choices: vec![Choice {
                    index: 0,
                    message: ResponseMessage {
                        role: "assistant".to_string(),
                        content: final_content,
                        reasoning_content: reasoning,
                        tool_calls,
                    },
                    finish_reason: Some(finish_reason.to_string()),
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            };
            Ok(Json(response).into_response())
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": { "message": e, "type": "internal_error" }
            })),
        )),
    }
}

/// Handle streaming response
async fn handle_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    request_id: String,
    created: i64,
    model_name: String,
    image: Option<image::DynamicImage>,
    params: GenerateParams,
    tools: Option<Vec<qwen_core::Tool>>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    // Collect all tokens in a blocking task, then stream them
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(256);

    let req_id = request_id.clone();
    let model = model_name.clone();
    let enable_thinking = params.enable_thinking;

    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            let mut pipeline = state_clone.pipeline.lock().await;
            let tools_ref = tools.as_deref();

            let result = pipeline.generate(
                &req.messages,
                tools_ref,
                image.as_ref(),
                &params,
                |token| {
                    let _ = tx.blocking_send(token.to_string());
                    StreamAction::Continue
                },
            );

            if let Err(e) = result {
                tracing::error!("Generation error: {}", e);
            }
        });
    });

    let stream = async_stream::stream! {
        // First chunk: role
        let first = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_name.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                    reasoning_content: None,
                },
                finish_reason: None,
            }],
        };
        yield Ok::<_, std::convert::Infallible>(
            Event::default().data(serde_json::to_string(&first).unwrap())
        );

        let mut rx = rx;
        let mut in_thinking = enable_thinking;
        let mut thinking_started = false;

        while let Some(token) = rx.recv().await {
            // Detect thinking transitions
            let (reasoning_content, content) = if in_thinking {
                if token.contains("</think>") {
                    in_thinking = false;
                    let parts: Vec<&str> = token.splitn(2, "</think>").collect();
                    let think_part = if !parts[0].is_empty() {
                        Some(parts[0].to_string())
                    } else {
                        None
                    };
                    let content_part = if parts.len() > 1 && !parts[1].is_empty() {
                        Some(parts[1].to_string())
                    } else {
                        None
                    };
                    // Send thinking part first if any
                    if let Some(tp) = think_part {
                        let chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_name.clone(),
                            choices: vec![StreamChoice {
                                index: 0,
                                delta: StreamDelta {
                                    role: None,
                                    content: None,
                                    reasoning_content: Some(tp),
                                },
                                finish_reason: None,
                            }],
                        };
                        yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                    }
                    (None, content_part)
                } else {
                    // Check for <think> tag at start
                    if !thinking_started && token.contains("<think>") {
                        thinking_started = true;
                        let after = token.splitn(2, "<think>").nth(1).unwrap_or("").to_string();
                        if after.is_empty() {
                            continue;
                        }
                        (Some(after), None)
                    } else {
                        (Some(token), None)
                    }
                }
            } else {
                (None, Some(token))
            };

            if reasoning_content.is_some() || content.is_some() {
                let chunk = ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_name.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: StreamDelta {
                            role: None,
                            content,
                            reasoning_content,
                        },
                        finish_reason: None,
                    }],
                };
                yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
            }
        }

        // Final chunk with finish_reason
        let final_chunk = ChatCompletionChunk {
            id: req_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };
        yield Ok(Event::default().data(serde_json::to_string(&final_chunk).unwrap()));
        yield Ok(Event::default().data("[DONE]".to_string()));
    };

    Ok(Sse::new(stream).into_response())
}

/// GET /v1/models
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Json<ModelsResponse> {
    let created = chrono::Utc::now().timestamp();
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created,
            owned_by: "local".to_string(),
        }],
    })
}

/// Parse <think>...</think> blocks from output
fn parse_thinking_content(output: &str, enable_thinking: bool) -> (Option<String>, String) {
    if !enable_thinking {
        return (None, output.to_string());
    }

    if let Some(think_end) = output.find("</think>") {
        let think_start = output.find("<think>").map(|i| i + 7).unwrap_or(0);
        let reasoning = output[think_start..think_end].trim().to_string();
        let content = output[think_end + 8..].trim().to_string();
        (
            if reasoning.is_empty() { None } else { Some(reasoning) },
            content,
        )
    } else {
        (None, output.to_string())
    }
}

/// Parse tool calls from Qwen's XML format
fn parse_tool_calls(content: &str) -> Option<Vec<ResponseToolCall>> {
    if !content.contains("<tool_call>") {
        return None;
    }

    let mut calls = Vec::new();

    for block in content.split("<tool_call>").skip(1) {
        if let Some(end) = block.find("</tool_call>") {
            let call_content = &block[..end];

            // Parse <function=name>...</function>
            if let Some(func_start) = call_content.find("<function=") {
                let after_eq = &call_content[func_start + 10..];
                if let Some(name_end) = after_eq.find('>') {
                    let func_name = after_eq[..name_end].to_string();

                    // Parse parameters
                    let mut args = serde_json::Map::new();
                    let func_body = &after_eq[name_end + 1..];
                    for param_block in func_body.split("<parameter=").skip(1) {
                        if let Some(pname_end) = param_block.find('>') {
                            let param_name = param_block[..pname_end].to_string();
                            if let Some(pval_end) = param_block.find("</parameter>") {
                                let param_value = param_block[pname_end + 1..pval_end]
                                    .trim()
                                    .to_string();
                                // Try to parse as JSON, fall back to string
                                let json_val = serde_json::from_str(&param_value)
                                    .unwrap_or(serde_json::Value::String(param_value));
                                args.insert(param_name, json_val);
                            }
                        }
                    }

                    calls.push(ResponseToolCall {
                        id: format!("call_{}", uuid::Uuid::new_v4()),
                        call_type: "function".to_string(),
                        function: ResponseFunctionCall {
                            name: func_name,
                            arguments: serde_json::to_string(&serde_json::Value::Object(args))
                                .unwrap_or_default(),
                        },
                    });
                }
            }
        }
    }

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// Strip <tool_call>...</tool_call> blocks from content
fn strip_tool_calls(content: &str) -> String {
    let mut result = content.to_string();
    while let Some(start) = result.find("<tool_call>") {
        if let Some(end) = result.find("</tool_call>") {
            result = format!(
                "{}{}",
                &result[..start],
                &result[end + 12..]
            );
        } else {
            break;
        }
    }
    result.trim().to_string()
}
