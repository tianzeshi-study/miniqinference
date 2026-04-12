# MiniQ Inference (miniqinference) v0.1.1

MiniQ Inference is a high-performance, local inference toolkit for **Qwen2-VL** and other multimodal models. It is built with **Rust** (using ONNX Runtime via the `ort` crate) and provides a multi-layer stack:

- **Core**: High-performance Rust inference engine with image preprocessing and chat templating.
- **CLI**: A command-line tool for local chat with vision support.
- **Server**: OpenAI-compatible API server (Axum-based) with streaming and tool calling support.
- **Python**: High-speed Python bindings for integrating MiniQ into your Python apps.

## Features

- 🏎️ **Optimized Inference**: Leverages ONNX Runtime with support for various backends.
- 🖼️ **Vision Support**: Native preprocessing for multimodal inputs (Qwen2-VL).
- 🧠 **Thinking Mode**: Built-in support for models with chain-of-thought capabilities.
- 🛠️ **Tool Calling**: Server support for local tool calling.
- ⚡ **Performance Profiling**: Built-in timing and memory usage monitoring (enable with `RUST_LOG=debug`).

## Installation

```bash
pip install miniqinference[all]
```

## Quick Start (CLI)

```bash
# Load model from a directory and chat
miniqwen-cli --model-dir ./models --prompt "Describe this image" --image ./test.jpg
```

## API Server

```bash
miniqwen-server --model-dir ./models --port 8000
```
Then use any OpenAI-compatible client.

## Performance Monitoring

Run with `RUST_LOG=debug` to see detailed execution time and memory consumption:
- Prefill duration
- Decoding speed (tokens/s and ms/token)
- Process memory usage (RSS)
