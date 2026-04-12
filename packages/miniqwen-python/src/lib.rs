use pyo3::prelude::*;
use miniqwen_core::config::QwenConfig;
use miniqwen_core::model::QwenEngine;
use tokenizers::Tokenizer;
use std::path::PathBuf;
use ort::value::DynValue;

#[pyclass]
struct Qwen {
    engine: QwenEngine,
    tokenizer: Tokenizer,
    config: QwenConfig,
}

#[pymethods]
impl Qwen {
    #[new]
    fn new(model_dir: String) -> PyResult<Self> {
        let model_path = PathBuf::from(model_dir);
        let config = QwenConfig::from_file(model_path.join("config.json"))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load config: {}", e)))?;
        
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load tokenizer: {}", e)))?;

        let engine = QwenEngine::new(&model_path, config.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to init engine: {}", e)))?;

        Ok(Qwen { engine, tokenizer, config })
    }

    fn generate(&mut self, prompt: String, max_tokens: usize) -> PyResult<String> {
        let encoding = self.tokenizer.encode(prompt.as_str(), true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Tokenization failed: {}", e)))?;
        
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mut kv_cache: Vec<DynValue> = self.engine.create_empty_kv_cache()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create KV cache: {}", e)))?;

        let eos_id = self.config.get_eos_id();
        
        let mut next_token = self.engine.forward(
            input_ids.clone(),
            &mut kv_cache,
            (0..input_ids.len() as i64).collect(),
            input_ids.len(),
            true
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Forward failed: {}", e)))?;

        let mut current_pos = input_ids.len() as i64;
        let mut result = String::new();

        for _ in 0..max_tokens {
            if next_token == eos_id {
                break;
            }

            let output_text = self.tokenizer.decode(&[next_token], true)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Decoding failed: {}", e)))?;
            result.push_str(&output_text);

            let input_ids = vec![next_token as i64];
            next_token = self.engine.forward(
                input_ids.clone(),
                &mut kv_cache,
                vec![current_pos],
                current_pos as usize + 1,
                false
            ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Forward failed: {}", e)))?;
            current_pos += 1;
        }

        Ok(result)
    }
}

#[pymodule]
fn miniqwen_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Qwen>()?;
    Ok(())
}
