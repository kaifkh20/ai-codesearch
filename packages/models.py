import os
from huggingface_hub import snapshot_download
from llama_cpp import Llama

# Paths where models will be stored
MODEL_DIR = "./models"
MODELS = {
    "phi3": {
        "repo_id": "microsoft/phi-3-mini-4k-instruct-gguf",
        "filename": "phi-3-mini-4k-instruct-q4_k_m.gguf",
    },
    "starcoder2": {
        "repo_id": "bigcode/starcoder2-3b-gguf",
        "filename": "starcoder2-3b-q4_k_m.gguf",
    }
}

_loaded_models = {}  # Cache for loaded models


def _download_model(model_key):
    """Download model if not already present."""
    model_info = MODELS[model_key]
    local_path = os.path.join(MODEL_DIR, model_key)

    # Check if file already exists
    file_path = os.path.join(local_path, model_info["filename"])
    if os.path.exists(file_path):
        return file_path

    print(f"Downloading {model_key} model from Hugging Face...")
    snapshot_download(
        repo_id=model_info["repo_id"],
        allow_patterns=model_info["filename"],
        local_dir=local_path
    )
    return file_path


def _load_model(model_key):
    """Load model with llama-cpp and cache it."""
    if model_key in _loaded_models:
        return _loaded_models[model_key]

    model_path = _download_model(model_key)
    print(f"Loading {model_key} model from {model_path}...")

    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4
    )

    _loaded_models[model_key] = llm
    return llm


def generate(model_key, prompt, max_tokens=256):
    """Generate text from a given model and prompt."""
    llm = _load_model(model_key)
    output = llm(prompt, max_tokens=max_tokens, stop=["</s>"])
    return output["choices"][0]["text"].strip()


# --- Example usage ---
if __name__ == "__main__":
    print("\n--- Phi-3 Mini (query rewrite) ---")
    print(generate("phi3", "Rewrite: find functions with nested loops"))

    print("\n--- StarCoder2 (summarization) ---")
    print(generate("starcoder2", "Summarize this function:\n\ndef add(a,b): return a+b"))

