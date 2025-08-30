import os
import ast
import json
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer,CrossEncoder
from sentence_transformers.util import cos_sim

from packages import bug
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
def generate_embeddings(chunks, index_file="index.json", batch_size=1, vector_mappings=None):
    if not chunks:
        print("No chunks to generate embeddings for.")
        return {}
    
    model = load_model()

    if vector_mappings is None:
        vector_mappings = {}
    
    print("Generating embeddings")
    
    # Group chunks by path
    chunks_by_path = {}
    for chunk in chunks:
        path = chunk[0]
        if path not in chunks_by_path:
            chunks_by_path[path] = []
        chunks_by_path[path].append(chunk)
    
    for path, path_chunks in chunks_by_path.items():
        # Skip if path already exists in vector_mappings
        if path in vector_mappings:
            print(f"Skipping path: {path} (already indexed)")
            continue
            
        print(f"Processing path: {path}")
        vector_mappings[path] = {}
        
        # Process chunks for this path in batches
        for i in range(0, len(path_chunks), batch_size):
            batch = path_chunks[i:i+batch_size]
            batch_codes = [code for _, _, _, _, code in batch]
            batch_embeddings = model.encode(batch_codes)
            
            for j, (_, fn_name, start, end, code) in enumerate(batch):
                func_key = f"{fn_name}_{start}_{end}"  # unique key within the path
                issues = bug.rules_bug(code)
                vector_mappings[path][func_key] = {
                    "embedding": batch_embeddings[j].tolist(),
                    "code": code,
                    "metadata": {
                        "function_name": fn_name, 
                        "start_line": start, 
                        "end_line": end,
                        "issues" : issues
                    }
                }

    # Save to cache
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(vector_mappings, f, indent=2)

    print(f"Generated embeddings for {sum(len(funcs) for funcs in vector_mappings.values())} functions across {len(vector_mappings)} files.")
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
def cosine_sim_cal(query_embed, vector_mappings, top_k=5):
    scores = Counter()
    query_embed = query_embed.astype(np.float32)
    
    # Flatten the nested structure for similarity calculation
    for path, functions in vector_mappings.items():
        for func_key, data in functions.items():
            embedding = np.array(data["embedding"], dtype=np.float32)
            similarity = cos_sim(query_embed, embedding)
            # Create unique key combining path and function
            combined_key = f"{path}::{func_key}"
            scores[combined_key] = float(similarity.item())
    
    return scores.most_common(top_k)

# --- Union Score ----
def union_score(cosine_scores, query):
    resulted_score = search_response(cosine_scores, query)
    return resulted_score

# --- Search response with keyword matching ---
def search_response(cosine_scores, query, alpha=0.7):
    """Literal match in code or function name."""
    query_lower = query.lower()
    words = query_lower.split()
    
    vector_mappings = load_embeddings()
    enhanced_scores = {}
    
    for combined_key, cosine_score in cosine_scores:
        # Parse the combined key
        path, func_key = combined_key.split("::")
        
        if path in vector_mappings and func_key in vector_mappings[path]:
            func_data = vector_mappings[path][func_key]
            code = func_data['code']
            code_lower = code.lower()
            fn_name_lower = func_data['metadata']['function_name'].lower()
            
            code_word_matches = sum(1 for word in words if word in code_lower)
            fn_word_matches = sum(1 for word in words if word in fn_name_lower)
            
            keyword_count = sum(code_lower.count(word) for word in words)
            keyword_score = keyword_count / len(words) if len(words) > 0 else 0
            
            final_score = alpha * cosine_score + (1 - alpha) * keyword_score
            enhanced_scores[combined_key] = {
                'score': final_score,
                'path': path,
                'func_key': func_key,
                'code': code,
                'metadata': func_data['metadata']
            }
    
    return enhanced_scores

# --- Cross Encoder ---
def cross_encoder(query, union_scores):
    """Re-rank results using cross-encoder for better relevance."""
    if not union_scores:
        return {}
    
    to_predict = [(query, data['code']) for (combined_key, data) in union_scores.items()]
    
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    
    predicted_scores = model.predict(to_predict)
    
    enhanced_scores = {}
    for idx, (combined_key, data) in enumerate(union_scores.items()):
        cross_score = float(predicted_scores[idx])
        original_score = data['score']
        
        # Weight combination: 0.6 cross-encoder + 0.4 original hybrid score
        final_score = 0.6 * cross_score + 0.4 * original_score
        
        enhanced_scores[combined_key] = {
            'score': final_score,
            'cross_score': cross_score,
            'original_score': original_score,
            'path': data['path'],
            'func_key': data['func_key'],
            'code': data['code'],
            'metadata': data['metadata']
        }
    
    return enhanced_scores


# --- Ranking ---
def ranking(scores, top_k=5):
    results = []
    # Sort by score in descending order
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    for i, (combined_key, data) in enumerate(sorted_scores[:top_k]):
        results.append({
            "key": combined_key,
            "score": data['score'],
            "path": data['path'],
            "code": data['code'],
            "metadata": data['metadata']
        })
    
    return results


# --- Formatting output ---
def format_response(results,bug_report=False):
    if not results:
        print("No matches found.")
        return
    print("Found in:")
    for result in results:
        meta = result["metadata"]
        if bug_report:
            print(f" - {result['path']}:{meta['start_line']}-{meta['end_line']} (function: {meta['function_name']}) | Score: {result['score']:.3f}\n - Issues:{meta['issues']}")
        else:
            print(f" - {result['path']}:{meta['start_line']}-{meta['end_line']} (function: {meta['function_name']}) | Score: {result['score']:.3f}")

# --- Full search pipeline ---
def search(folder, query, top_k=5, batch_size=2, max_lines=2000, index_file="index.json",bugs=False):
    # 1. Load cached embeddings if exist
    vector_mappings = load_embeddings(index_file)
    
    # 2. Read files and generate embeddings in batches
    code_chunks = read_files_python(folder, max_lines=max_lines)
    vector_mappings = generate_embeddings(
        chunks=code_chunks,
        index_file=index_file,
        batch_size=batch_size,
        vector_mappings=(vector_mappings if len(vector_mappings) > 0 else None)
    )
    
    if not vector_mappings:
        print("No embeddings available, exiting.")
        return
    
    # 3. Generate query embedding
    query_embedding = generate_query_embeddings(query)
    
    # 4. Cosine similarity
    cosine_scores = cosine_sim_cal(query_embedding, vector_mappings, top_k=top_k*2)  # Get more for hybrid scoring
    
    # 5. Apply hybrid scoring
    union_scores = union_score(cosine_scores, query)
    
    #6 Apply cross encoding
    final_scores = cross_encoder(query, union_scores)
    
    # 7. Rank top K results
    ranked_results = ranking(final_scores, top_k=top_k)    
    # 7. Show output
    format_response(ranked_results,bug_report=bugs)
