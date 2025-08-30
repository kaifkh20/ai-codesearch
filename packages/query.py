import os
import ast
import json
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# --- Extract functions ---
def extract_functions_from_file(path, max_lines=500):
    """Extract functions from a Python file. Skip very large files."""
    chunks = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        if len(code.splitlines()) > max_lines:
            # skip very large files to save memory
            return chunks
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start = node.lineno
                end = max([n.lineno for n in ast.walk(node) if hasattr(n,"lineno")], default=start)
                lines = code.splitlines()[start-1:end]
                chunks.append((path, node.name, start, end, "\n".join(lines)))
        if not chunks:  # fallback: entire file as chunk
            chunks.append((path, "<file>", 1, len(code.splitlines()), code))
    except Exception as e:
        print(f"Could not parse {path}: {e}")
    return chunks

def read_files_python(folder, max_lines=500):
    """Read all Python files in folder, skipping large files."""
    chunks = []
    for root, sub_dirs, files in os.walk(folder):
        if ".venv" in sub_dirs:
            sub_dirs.remove(".venv")
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                chunks.extend(extract_functions_from_file(path, max_lines=max_lines))
    return chunks

# --- Model loading ---
def load_model():
    return SentenceTransformer("jinaai/jina-embeddings-v2-base-code", trust_remote_code=True)

# --- Embedding generation with batching ---
def generate_embeddings(chunks, index_file="index.json", batch_size=1):
    if not chunks:
        print("No chunks to generate embeddings for.")
        return {}

    model = load_model()
    vector_mappings = {}

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_codes = [code for _, _, _, _, code in batch]
        batch_embeddings = model.encode(batch_codes)

        for j, (path, fn_name, start, end, code) in enumerate(batch):
            key = f"fn_{i+j+1}"
            vector_mappings[key] = {
                "embedding": batch_embeddings[j].tolist(),
                "code": code,
                "metadata": {"path": path, "function_name": fn_name, "start_line": start, "end_line": end}
            }

    # save to cache
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(vector_mappings, f, indent=2)

    print(f"Generated embeddings for {len(vector_mappings)} chunks.")
    return vector_mappings

# --- Load cached embeddings ---
def load_embeddings(index_file="index.json"):
    if not os.path.exists(index_file):
        return {}
    with open(index_file, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Query embedding ---
def generate_query_embeddings(query):
    model = load_model()
    return model.encode([query])[0]

# --- Cosine similarity ---
def cosine_sim_cal(query_embed, vector_mappings):
    scores = Counter()
    query_embed = query_embed.astype(np.float32)
    for key, data in vector_mappings.items():
        embedding = np.array(data["embedding"],dtype=np.float32)
        similarity = cos_sim(query_embed, embedding)
        scores[key] = float(similarity.item())
    return scores

# --- Ranking ---
def ranking(scores, top_k=5, vector_mappings=None):
    results = []
    for key, score in scores.most_common(top_k):
        if vector_mappings and key in vector_mappings:
            results.append({
                "key": key,
                "score": score,
                "data": vector_mappings[key]
            })
    return results

# --- Search ---
def search_response(chunks, query):
    """Literal match in code or function name."""
    query_lower = query.lower()
    results = []
    for path, fn_name, start, end, code in chunks:
        if query_lower in code.lower() or query_lower in fn_name.lower():
            results.append((path, fn_name, start, end, code))
    return results

# --- Formatting output ---
def format_response(results):
    if not results:
        print("No matches found.")
        return
    print("Found in:")
    for result in results:
        meta = result["data"]["metadata"]
        print(f" - {meta['path']}:{meta['start_line']}-{meta['end_line']} (function: {meta['function_name']}) | Score: {result['score']:.3f}")

# --- Full search pipeline ---
def search(folder, query, top_k=5, batch_size=2, max_lines=500, index_file="index.json"):
    # 1. Load cached embeddings if exist
    vector_mappings = load_embeddings(index_file)
    
    if vector_mappings:
        print(f"Loaded {len(vector_mappings)} cached embeddings.")
    else:
        print("No cached embeddings found, generating embeddings...")
        # read files and generate embeddings in batches
        code_chunks = read_files_python(folder, max_lines=max_lines)
        vector_mappings = generate_embeddings(code_chunks, index_file=index_file, batch_size=batch_size)
    
    if not vector_mappings:
        print("No embeddings available, exiting.")
        return
    
    # 2. Generate query embedding
    query_embedding = generate_query_embeddings(query)
    
    # 3. Cosine similarity
    scores = cosine_sim_cal(query_embedding, vector_mappings)
    
    # 4. Rank top K results
    ranked_results = ranking(scores, top_k=top_k, vector_mappings=vector_mappings)
    
    # 5. Show output
    format_response(ranked_results)
