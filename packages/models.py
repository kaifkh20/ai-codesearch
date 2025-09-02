import os
from huggingface_hub import hf_hub_download, list_repo_files
from llama_cpp import Llama

# Paths where models will be stored
MODEL_DIR = "./models"
MODELS = {
    "phi3": {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
    },
    "starcoder2": {
        "repo_id": "bartowski/starcoder2-3b-GGUF",
        "filename": "starcoder2-3b-Q4_K_M.gguf",
    }
}

loaded_models = {}  # Cache for loaded models

def list_available_files(repo_id):
    """List all files in a HuggingFace repository."""
    try:
        files = list_repo_files(repo_id)
        print(f"Available files in {repo_id}:")
        for file in files:
            if file.endswith('.gguf'):
                print(f"  - {file}")
        return files
    except Exception as e:
        print(f"Error listing files in {repo_id}: {e}")
        return []

def download_model(model_key):
    """Download model if not already present using hf_hub_download."""
    model_info = MODELS[model_key]
    local_path = os.path.join(MODEL_DIR, model_key)
    
    # Create directories if they don't exist
    os.makedirs(local_path, exist_ok=True)
    
    # Check if file already exists
    file_path = os.path.join(local_path, model_info["filename"])
    if os.path.exists(file_path):
        print(f"Model {model_key} already exists at {file_path}")
        return file_path
    
    print(f"Checking available files for {model_key}...")
    available_files = list_available_files(model_info["repo_id"])
    
    # Check if the expected file exists in the repo
    if model_info["filename"] not in available_files:
        print(f"ERROR: File {model_info['filename']} not found in repository!")
        print("Available .gguf files:")
        for file in available_files:
            if file.endswith('.gguf'):
                print(f"  - {file}")
        return None
    
    print(f"Downloading {model_key} model from Hugging Face...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["filename"],
            local_dir=local_path,
        )
        print(f"Successfully downloaded {model_key} to {downloaded_path}")
        
        # The actual file path where it was downloaded
        actual_file_path = os.path.join(local_path, model_info["filename"])
        if os.path.exists(actual_file_path):
            return actual_file_path
        else:
            print(f"File downloaded but not found at expected location: {actual_file_path}")
            print(f"Downloaded to: {downloaded_path}")
            return downloaded_path
            
    except Exception as e:
        print(f"Error downloading {model_key}: {e}")
        raise

def load_model(model_key):
    """Load model with llama-cpp and cache it."""
    if model_key in loaded_models:
        print(f"Using cached model: {model_key}")
        return loaded_models[model_key]
    
    model_path = download_model(model_key)
    
    if model_path is None:
        raise ValueError(f"Could not download model {model_key}")
    
    # Double-check the file exists after download
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found after download: {model_path}")
    
    print(f"Loading {model_key} model from {model_path}...")
    print(f"File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=os.cpu_count() or 4,
            verbose=False  # Reduce noise
        )
        loaded_models[model_key] = llm
        print(f"Successfully loaded {model_key}")
        return llm
    except Exception as e:
        print(f"Error loading {model_key}: {e}")
        raise

def generate(model_key, prompt, max_tokens=256):
    """Generate text from a given model and prompt."""
    llm = load_model(model_key)
    output = llm(prompt, max_tokens=max_tokens, stop=["</s>"])
    return output["choices"][0]["text"].strip()

# --- Example usage ---
if __name__ == "__main__":
    # Add some debug info
    print(f"Model directory: {os.path.abspath(MODEL_DIR)}")
    print(f"Directory exists: {os.path.exists(MODEL_DIR)}")
    
    # Test with just phi3 first
    try:
        print("\n--- Testing Phi-3 Mini ---")
        result = generate("phi3", "Rewrite: find functions with nested loops")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error with phi3: {e}")
        print("Let's check what files are actually available...")
        list_available_files("microsoft/Phi-3-mini-4k-instruct-gguf")
