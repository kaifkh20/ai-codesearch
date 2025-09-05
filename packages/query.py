import os

import json
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer,CrossEncoder
from sentence_transformers.util import cos_sim

import faiss
from packages import bug
from packages import parser
#from packages import models

from google import genai
from dotenv import load_dotenv
import os
load_dotenv()


API_KEY = os.getenv("GEMINI_API")

# Global singletons
embedding_model = SentenceTransformer("jinaai/jina-embeddings-v2-base-code", trust_remote_code=True)
cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

def generate_consistent_key(chunk):
    """Generate a consistent key for both FAISS mapping and vector mapping"""
    # Use the chunk name (which includes _part1, _part2 for split chunks) + start/end for uniqueness
    name = chunk.get("fq_name", "unknown")
    start = chunk.get("start", 0)
    end = chunk.get("end", 0)
    return f"{name}_{start}_{end}"

# --- Add to FAISS
def add_to_faiss(batch, path, faiss_file="index.faiss", vector_file="vector_map.json"):
    # Generate embeddings for all chunks in the batch
    codes = [f'Summary:{chunk.get("summary"," ")}Code:{chunk["code"]}' for chunk in batch]
    embeddings = embedding_model.encode(codes, batch_size=2, convert_to_numpy=True, normalize_embeddings=True)
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

    # Update mapping with consistent key generation
    for j, chunk in enumerate(batch):
        faiss_id = batch_indices[j]
        func_key = generate_consistent_key(chunk)

        existing_mapping[str(faiss_id)] = {
            "path": path,
            "func_key": func_key,
        }

    # Save index + mapping once
    faiss.write_index(index, faiss_file)

    return batch_indices

# --- Embedding generation with streaming ---
def generate_embeddings(chunks, index_file="vector_index.json", batch_size=2, vector_mappings=None):
    if not chunks:
        print("No chunks to generate embeddings for.")
        return {}

    if vector_mappings is None:
        vector_mappings = {}


    # Group by path so we only process per file
    chunks_by_path = {}
    for chunk in chunks:
        chunks_by_path.setdefault(chunk["path"], []).append(chunk)

    for path, path_chunks in chunks_by_path.items():
        if path in vector_mappings:
            print(f"Skipping {path} (already indexed)")
            continue

        print(f"Processing {path} with {len(path_chunks)} chunks...")

        # Process in smaller batches to avoid OOM
        faiss_ids = []
        for i in range(0, len(path_chunks), batch_size):
            mini_batch = path_chunks[i:i + batch_size]
            faiss_ids.extend(add_to_faiss(mini_batch, path))

        # Initialize per-file mapping
        vector_mappings[path] = {}

        # Save results incrementally with consistent key generation
        for j, chunk in enumerate(path_chunks):
            func_key = generate_consistent_key(chunk)

            if chunk["language"] == 'python':
                issues = bug.rules_bug(chunk["code"])
            else:
                issues = []

            vector_mappings[path][func_key] = {
                "embedding": faiss_ids[j],  # FAISS id
                "code": chunk["code"],
                "metadata": {
                    "function_name": chunk.get("name", "unknown"),
                    "fq_name": chunk.get("fq_name", chunk.get("name", "unknown")),
                    "category": chunk["category"],
                    "language": chunk["language"],
                    "start_line": chunk["start"],
                    "end_line": chunk["end"],
                    "node_type": chunk.get("node_type", "unknown"),
                    "issues": issues,
                }
            }

        # Save after each file to avoid memory buildup
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(vector_mappings, f, indent=2)

    return vector_mappings


# --- Load cached embeddings ---
def load_embeddings(index_file="vector_index.json"):
    if not os.path.exists(index_file):
        return {}
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading embeddings: {e}")
        return {}

# --- Query embedding ---
def generate_query_embeddings(query):
    model = embedding_model
    return model.encode(query)

def cosine_sim_cal(query_embed,vector_mappings,faiss_file="index.faiss", top_k=10):
    
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

    if index.ntotal == 0:
        print("FAISS index is empty!")
        return Counter()

    distances, indices = index.search(query_embed, top_k)
    
    # Load vector mapping with error handling
    try:
        with open("vector_map.json",'r') as f:
            embed_map = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading vector map: {e}")
        return Counter()

    scores = Counter()
    for idx, distance in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        mapping = embed_map.get(str(idx))
        if mapping is None:
            print(f"Warning: No mapping found for index {idx}")
            continue
        combined_key = f"{mapping['path']}::{mapping['func_key']}"
        scores[combined_key] = float(distance)

    print(f"Found {len(scores)} matches from cosine similarity")
    return scores

# --- Union Score ----
def union_score(cosine_scores, query):
    resulted_score = search_response(cosine_scores, query)
    return resulted_score

#--- Search Response

def search_response(cosine_scores, query, alpha=0.7):
    """
    Hybrid scoring: cosine + keyword overlap + category + name/code boosts.
    Boosts:
      - function name match: +2
      - code match: +0.5
      - slight mention (any match at all): +0.2
    """
    if not cosine_scores:
        print("No cosine scores to process")
        return {}
        
    query_lower = ",".join(query).lower()
    words = query_lower.split(',')
    
    vector_mappings = load_embeddings()
    enhanced_scores = {}
    
    for combined_key, cosine_score in cosine_scores.items():
        # Parse the combined key
        
        try:
            path, func_key = combined_key.split("::")
        except ValueError:
            print(f"Warning: Invalid combined key format: {combined_key}")
            continue
        
        if path in vector_mappings and func_key in vector_mappings[path]:
            func_data = vector_mappings[path][func_key]
            code = func_data['code']
            code_lower = code.lower()
            fn_name_lower = func_data['metadata']['function_name'].lower()
            fq_name_lower = func_data['metadata']['fq_name'].lower()
            category = func_data['metadata'].get("category", "function")
            
            # --- keyword-based signals ---
            code_word_matches = sum(0.3 for word in words if word in code_lower)
            fn_word_matches = sum(0.7 for word in words if word in fn_name_lower)
            fq_word_matches = sum(0.5 for word in words if word in fq_name_lower)
            
            # Slight mention bonus: if any word is in name or code
            slight_mention = 0.5 if any(
                (word in fn_name_lower or word in fq_name_lower) for word in words
            ) else 0.0
            slight_mention += 0.2 if any(
                (word in code_lower) for word in words
            ) else 0.0
            
            keyword_score = (code_word_matches + fn_word_matches + fq_word_matches + slight_mention)
            
            # --- hybrid score ---
            final_score = alpha * cosine_score + (1 - alpha) * keyword_score
            
            # --- category boosts ---
            if "class" in query_lower or "object" in query_lower:
                if category == "class":
                    final_score *= 2
            if "method" in query_lower or "member" in query_lower:
                if category == "method":
                    final_score *= 1.1
            if "function" in query_lower or "def" in query_lower:
                if category == "function":
                    final_score *= 1.05
            
            enhanced_scores[combined_key] = {
                'score': final_score,
                'path': path,
                'func_key': func_key,
                'code': code,
                'metadata': func_data['metadata']
            }
        else:
            print(f"Warning: Could not find data for {combined_key}")
    
    print(f"Enhanced scores: {len(enhanced_scores)} results")
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


# --- Ranking with function-level collapsing ---
def ranking(scores, top_k=10):
    if not scores:
        return []
        
    grouped = {}
    
    # Group results by fq_name (function) instead of individual chunks
    for combined_key, data in scores.items():
        fq_name = data['metadata'].get('fq_name', data['metadata']['function_name'])
        path = data['path']
        group_key = f"{path}::{fq_name}"

        if group_key not in grouped:
            grouped[group_key] = {
                "path": path,
                "func_key": fq_name,
                "code": data["code"],  # start with one chunk
                "metadata": data["metadata"].copy(),
                "score": data["score"],
                "chunks": [data["code"]],
                "chunk_scores": [data["score"]],
            }
        else:
            # Update best score across chunks and merge code
            if data["score"] > grouped[group_key]["score"]:
                grouped[group_key]["score"] = data["score"]
                grouped[group_key]["metadata"] = data["metadata"].copy()
            
            # Merge code snippets for context
            grouped[group_key]["chunks"].append(data["code"])
            grouped[group_key]["chunk_scores"].append(data["score"])
            
            # Update start/end line boundaries to encompass all chunks
            grouped[group_key]["metadata"]["start_line"] = min(
                grouped[group_key]["metadata"]["start_line"], data["metadata"]["start_line"]
            )
            grouped[group_key]["metadata"]["end_line"] = max(
                grouped[group_key]["metadata"]["end_line"], data["metadata"]["end_line"]
            )
    
    # Replace "code" with merged text for final output
    for g in grouped.values():
        # Sort chunks by their scores and combine them
        chunk_pairs = list(zip(g["chunks"], g["chunk_scores"]))
        chunk_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by score
        g["code"] = "\n\n# --- [CHUNK SEPARATOR] ---\n\n".join([chunk for chunk, _ in chunk_pairs])
        del g["chunks"]
        del g["chunk_scores"]

    # Sort by score in descending order
    sorted_scores = sorted(grouped.items(), key=lambda x: x[1]['score'], reverse=True)

    results = []
    for i, (group_key, data) in enumerate(sorted_scores[:top_k]):
        results.append({
            "key": group_key,
            "score": data['score'],
            "path": data['path'],
            "code": data['code'],
            "metadata": data['metadata']
        })

    return results



# --- Formatting output (grouped by function) ---
def format_response(results, bug_report=False):
    if not results:
        print("No matches found.")
        return

    print(f"Found {len(results)} matches:")
    for result in results:
        meta = result["metadata"]
        func_name = meta["function_name"]
        fq_name = meta.get("fq_name", func_name)
        start, end = meta["start_line"], meta["end_line"]
        score = result["score"]

        if bug_report:
            print(
                f" - {result['path']}:{start}-{end} "
                f"({meta['category']}: {fq_name}) | Score: {score:.3f} | \n Issues :"
            )
            for issue in meta.get('issues', []):
                print(f"\t- {issue}")
        else:
            print(
                f" - {result['path']}:{start}-{end} "
                f"({meta['category']} {fq_name}) | Score: {score:.3f}"
            )

# --- Query rewriter

def query_rewriter(raw_query):
    print("=== QUERY REWRITER START ===")
    try:
        client = genai.Client(api_key=API_KEY)
        prompt = f"Rewrite the following natural language query into developer keywords, and function names or class names that might appear in the codebase.Query:{raw_query};Output as a comma-separated list of terms."
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        # Clean up the split terms by stripping whitespace
        rewritten_terms = [term.strip() for term in response.text.split(',')]
        print("Re-written Query", rewritten_terms)
        print("=== QUERY REWRITER END ===")
        
        return rewritten_terms + raw_query.split(' ')
    except Exception as e:
        print(f"Query rewriter error: {e}")
        return raw_query.split(' ')

# --- Debug function to inspect index state
def debug_index_state():
    print("=== INDEX DEBUG ===")
    print(f"vector_index.json exists: {os.path.exists('vector_index.json')}")
    print(f"index.faiss exists: {os.path.exists('index.faiss')}")
    print(f"vector_map.json exists: {os.path.exists('vector_map.json')}")
    
    if os.path.exists('vector_index.json'):
        with open('vector_index.json', 'r') as f:
            data = json.load(f)
            print(f"vector_index.json has {len(data)} files")
            total_chunks = sum(len(file_data) for file_data in data.values())
            print(f"Total chunks in vector_index.json: {total_chunks}")
    
    if os.path.exists('index.faiss'):
        try:
            index = faiss.read_index('index.faiss')
            print(f"FAISS index has {index.ntotal} vectors")
        except Exception as e:
            print(f"Error reading FAISS index: {e}")
    
    if os.path.exists('vector_map.json'):
        with open('vector_map.json', 'r') as f:
            data = json.load(f)
            print(f"vector_map.json has {len(data)} mappings")
            # Show a sample mapping
            if data:
                sample_key = next(iter(data))
                print(f"Sample mapping: {sample_key} -> {data[sample_key]}")

# --- Full search pipeline ---
def search(folder, query, top_k=10, batch_size=2, max_lines=2000, index_file="vector_index.json", bugs=False, debug=False, force_rebuild=False):
    print(f"=== SEARCH START for query: '{query}' ===")
    
    # Force rebuild if requested
    if force_rebuild:
        print("Force rebuild requested...")
        force_rebuild_indices()
    
    if debug:
        print("Before search:")
        debug_index_state()
    
    # 1. Load cached embeddings if exist
    vector_mappings = load_embeddings(index_file)
    print(f"Loaded {len(vector_mappings)} files from cache")
    
    # 2. Read files and generate embeddings in batches
    code_chunks = parser.read_files(folder)
    print(f"Found {len(code_chunks)} code chunks from parser")
    
    if API_KEY:
        query_rewritten = query_rewriter(query)
    else:
        print("******API KEY IS NOT PROVIDED REWRITTEN QUERY IS TURNED OFF*****")
        query_rewritten = query.split(" ")

    vector_mappings = generate_embeddings(
        chunks=code_chunks,
        index_file=index_file,
        batch_size=batch_size,
        vector_mappings=(vector_mappings if len(vector_mappings) > 0 else None)
    )
    
    if not vector_mappings:
        print("No embeddings available, exiting.")
        return
    
    if debug:
        print("After embedding generation:")
        debug_index_state()
    
    # 3. Generate query embedding
    print("Generating query embedding...")
    query_embedding = generate_query_embeddings(query=query_rewritten)
    
    # 4. Cosine similarity
    print("Computing cosine similarity...")
    cosine_scores = cosine_sim_cal(query_embedding, vector_mappings, top_k=top_k*2)  # Get more for hybrid scoring
    
    if not cosine_scores:
        print("No cosine similarity matches found.")
        print("This might indicate index inconsistency. Try running with force_rebuild=True")
        if debug:
            print("After cosine similarity:")
            debug_index_state()
        return
    
    # 5. Apply hybrid scoring
    print("Applying hybrid scoring...")
    union_scores = union_score(cosine_scores, query_rewritten)
    
    if not union_scores:
        print("No union scores computed.")
        print("This might indicate index inconsistency. Try running with force_rebuild=True")
        return
    
    #6 Apply cross encoding
    print("Applying cross-encoder re-ranking...")
    final_scores = cross_encoder(query, union_scores)
    
    # 7. Rank top K results
    ranked_results = ranking(final_scores, top_k=top_k)    
    
    # 8. Show output
    format_response(ranked_results, bug_report=bugs)
    
    print("=== SEARCH END ===")
