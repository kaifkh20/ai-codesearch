import os
import ast
import json
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer,CrossEncoder
from sentence_transformers.util import cos_sim

import faiss
from packages import bug
from packages import parser

# Global singletons
embedding_model = SentenceTransformer("jinaai/jina-embeddings-v2-base-code", trust_remote_code=True)
cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

# --- Add to FAISS
def add_to_faiss(batch, path, faiss_file="index.faiss", vector_file="vector_map.json"):
    # Generate embeddings for all chunks in the batch
    codes = [chunk["code"] for chunk in batch]
    embeddings = embedding_model.encode(codes, batch_size=16, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]

    # Load or init FAISS index
    if os.path.exists(faiss_file):
        index = faiss.read_index(faiss_file)
        try:
            with open(vector_file, "r") as f:
                existing_mapping = json.load(f)
        except FileNotFoundError:
            existing_mapping = {}
    else:
        index = faiss.IndexFlatIP(dimension)
        existing_mapping = {}

    start_index = index.ntotal
    index.add(embeddings)
    batch_indices = list(range(start_index, index.ntotal))

    # Update mapping
    for j, chunk in enumerate(batch):
        faiss_id = batch_indices[j]
        fn_name = chunk["fq_name"] or chunk["name"]
        start = chunk["start"]
        end = chunk["end"]

        func_key = f"{fn_name}_{start}_{end}"
        existing_mapping[str(faiss_id)] = {
            "path": path,
            "func_key": func_key,
        }

    # Save index + mapping once
    faiss.write_index(index, faiss_file)
    with open(vector_file, "w", encoding="utf-8") as f:
        json.dump(existing_mapping, f, indent=2)

    return batch_indices

# --- Embedding generation with batching ---
def generate_embeddings(chunks, index_file="index.json", batch_size=2, vector_mappings=None):
    if not chunks:
        print("No chunks to generate embeddings for.")
        return {}

    if vector_mappings is None:
        vector_mappings = {}

    chunks_by_path = {}
    for chunk in chunks:
        chunks_by_path.setdefault(chunk["path"], []).append(chunk)

    for path, path_chunks in chunks_by_path.items():
        if path in vector_mappings:
            print(f"Skipping {path} (already indexed)")
            continue

        print(f"Processing {path} with {len(path_chunks)} chunks...")

        # Add all chunks for this file to FAISS once
        faiss_ids = add_to_faiss(path_chunks, path)

        vector_mappings[path] = {}
        for j, chunk in enumerate(path_chunks):
            fn_name = chunk["fq_name"] or chunk["name"]
            start = chunk["start"]
            end = chunk["end"]
            func_key = f"{fn_name}_{start}_{end}"
            issues = bug.rules_bug(chunk["code"])

            vector_mappings[path][func_key] = {
                "embedding": faiss_ids[j],  # FAISS id
                "code": chunk["code"],
                "metadata": {
                    "function_name": fn_name,
                    "category": chunk["category"],
                    "language": chunk["language"],
                    "start_line": start,
                    "end_line": end,
                    "issues": issues
                }
            }

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(vector_mappings, f, indent=2)

    return vector_mappings

# --- Load cached embeddings ---
def load_embeddings(index_file="index.json"):
    if not os.path.exists(index_file):
        return {}
    with open(index_file, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Query embedding ---
def generate_query_embeddings(query):
    model = embedding_model
    return model.encode([query])[0]

def cosine_sim_cal(query_embed,vector_mappings,faiss_file="index.faiss", top_k=5):
    
    # Ensure query_embed is 2D
    query_embed = np.array(query_embed, dtype=np.float32)
    if query_embed.ndim == 1:
        query_embed = query_embed.reshape(1, -1)

    # Normalize query embedding (cosine similarity)
    query_embed = query_embed / np.linalg.norm(query_embed, axis=1, keepdims=True)

    # Load FAISS index
    try:
        index = faiss.read_index(faiss_file)
        print(f"Loaded FAISS index with {index.ntotal} vectors")
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index: {e}")

    distances, indices = index.search(query_embed, top_k)
    
    with open("vector_map.json",'r') as f:
        embed_map = json.load(f)

    scores = Counter()
    for idx, distance in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        mapping =embed_map.get(str(idx))
        combined_key = f"{mapping['path']}::{mapping['func_key']}"
        scores[combined_key] = float(distance)

    return scores

# --- Union Score ----
def union_score(cosine_scores, query):
    resulted_score = search_response(cosine_scores, query)
    return resulted_score

#--- Search Response

def search_response(cosine_scores, query, alpha=0.7):
    """
    Hybrid scoring: cosine + keyword overlap + category awareness.
    Boosts classes if query hints at class/object.
    """
    query_lower = query.lower()
    words = query_lower.split()
    
    vector_mappings = load_embeddings()
    enhanced_scores = {}
    
    for combined_key, cosine_score in cosine_scores.items():
        # Parse the combined key
        path, func_key = combined_key.split("::")
        
        if path in vector_mappings and func_key in vector_mappings[path]:
            func_data = vector_mappings[path][func_key]
            code = func_data['code']
            code_lower = code.lower()
            fn_name_lower = func_data['metadata']['function_name'].lower()
            category = func_data['metadata'].get("category", "function")  # default
            
            # --- keyword-based signals ---
            code_word_matches = sum(1 for word in words if word in code_lower)
            fn_word_matches = sum(1.5 for word in words if word in fn_name_lower)
            
            keyword_count = sum(code_lower.count(word) for word in words)
            keyword_score = keyword_count / len(words) if len(words) > 0 else 0
            
            # --- hybrid score ---
            final_score = alpha * cosine_score + (1 - alpha) * keyword_score
            
            # --- category boosts ---
            if "class" in query_lower or "object" in query_lower:
                if category == "class":
                    final_score *= 1.2   # boost class results
            
            if "method" in query_lower or "member" in query_lower:
                if category == "method":
                    final_score *= 1.1   # boost methods slightly
            
            if "function" in query_lower or "def" in query_lower:
                if category == "function":
                    final_score *= 1.05  # boost functions slightly
            
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
    
    model = cross_model
    
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
            print(f" - {result['path']}:{meta['start_line']}-{meta['end_line']} ",f"({meta['category']}: {meta['function_name']}) | Score: {result['score']:.3f}")
        else:
            print(f" - {result['path']}:{meta['start_line']}-{meta['end_line']} ",f"({meta['category']} {meta['function_name']}) | Score: {result['score']:.3f}")

# --- Full search pipeline ---
def search(folder, query, top_k=5, batch_size=2, max_lines=2000, index_file="index.json",bugs=False):
    # 1. Load cached embeddings if exist
    vector_mappings = load_embeddings(index_file)
    
    # 2. Read files and generate embeddings in batches
    #code_chunks = read_files_python(folder, max_lines=max_lines)
    
    code_chunks = parser.read_files(folder)

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
