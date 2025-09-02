import sys
import os
import json
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language

from google import genai
from dotenv import load_dotenv
import os
import time

from packages import models

load_dotenv()


API_KEY = os.getenv("GEMINI_API")

class LanguageConfig:
    """Configuration for different programming languages"""
    
    LANGUAGES = {
        'python': {
            'extensions': ['.py'],
            'language_name': 'python',
            'function_types': ['function_definition'],
            'class_types': ['class_definition'],
            'method_types': ['function_definition'],  # Methods are also function_definition in Python
        },
        'javascript': {
            'extensions': ['.js', '.jsx', '.mjs'],
            'language_name': 'javascript',
            'function_types': ['function_declaration', 'function_expression', 'arrow_function'],
            'class_types': ['class_declaration'],
            'method_types': ['method_definition'],
        },
        'typescript': {
            'extensions': ['.ts', '.tsx'],
            'language_name': 'typescript',
            'function_types': ['function_declaration', 'function_signature', 'arrow_function'],
            'class_types': ['class_declaration'],
            'method_types': ['method_definition', 'method_signature'],
        },
        'java': {
            'extensions': ['.java'],
            'language_name': 'java',
            'function_types': ['method_declaration'],
            'class_types': ['class_declaration'],
            'method_types': ['method_declaration'],
        },
        'cpp': {
            'extensions': ['.cpp', '.cc', '.cxx', '.c++', '.hpp', '.h'],
            'language_name': 'cpp',
            'function_types': ['function_definition', 'function_declarator'],
            'class_types': ['class_specifier'],
            'method_types': ['function_definition'],
        },
        'c': {
            'extensions': ['.c', '.h'],
            'language_name': 'c',
            'function_types': ['function_definition', 'function_declarator'],
            'class_types': [],  # C doesn't have classes
            'method_types': [],
        },
        'rust': {
            'extensions': ['.rs'],
            'language_name': 'rust',
            'function_types': ['function_item'],
            'class_types': ['struct_item', 'enum_item'],
            'method_types': ['function_item'],  # Methods are also function_item in impl blocks
        },
        'go': {
            'extensions': ['.go'],
            'language_name': 'go',
            'function_types': ['function_declaration', 'method_declaration'],
            'class_types': ['type_declaration'],  # Go uses type declarations for structs
            'method_types': ['method_declaration'],
        }
    }
    
    @classmethod
    def get_language_for_file(cls, filepath):
        """Determine language based on file extension"""
        _, ext = os.path.splitext(filepath.lower())
        for lang_name, config in cls.LANGUAGES.items():
            if ext in config['extensions']:
                return lang_name, config
        return None, None

def load_index(index_path):
    """Load existing index.json file if it exists"""
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load index file {index_path}: {e}")
            return {}
    return {}

def save_index(index_data, index_path):
    """Save index data to index.json file"""
    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        print(f"Index saved to {index_path}")
    except IOError as e:
        print(f"Warning: Could not save index file {index_path}: {e}")

def should_parse_file(file_path, index_data, check_mtime=True):
    """
    Check if a file should be parsed based on index data.
    """
    # Normalize path for consistent comparison
    normalized_path = os.path.normpath(file_path)
    
    # If file is not in index, it should be parsed
    if normalized_path not in index_data:
        return True
    
    # If we're not checking modification time, skip parsing
    if not check_mtime:
        print(f"Skipping {file_path}: found in index")
        return False
    
    # Check if file was modified since last indexing
    try:
        file_mtime = os.path.getmtime(file_path)

        indexed_mtime = index_data[normalized_path].get('last_modified', 0)
        
        print("Normalized path",normalized_path)

        if file_mtime > indexed_mtime:
            print(f"Re-parsing {file_path}: file modified since last index")
            return True
        else:
            print(f"Skipping {file_path}: up to date in index")
            return False
            
    except (OSError, KeyError, TypeError):
        # If there's any error checking modification time, parse the file
        print(f"Re-parsing {file_path}: could not verify modification time")
        return True

def get_identifier_name(node, code, language_config=None):
    # --- Direct field name (works for Python, Java, etc.) ---
    name_node = node.child_by_field_name('name')
    if name_node:
        return code[name_node.start_byte:name_node.end_byte]

    # --- Shallow identifier check (works for many grammars) ---
    for child in node.children:
        if child.type in ('identifier', 'property_identifier', 'type_identifier', 'field_identifier'):
            return code[child.start_byte:child.end_byte]

    # --- Special handling for C/C++ function definitions ---
    if node.type == "function_definition":
        declarator = node.child_by_field_name("declarator")
        if declarator:
            ident = _find_identifier_recursive(declarator, code)
            if ident:
                return ident

    # --- Nothing found ---
    return "unknown"


def _find_identifier_recursive(node, code):
    """Recursively search for an identifier inside declarators."""
    if node.type == "identifier":
        return code[node.start_byte:node.end_byte]
    for child in node.children:
        result = _find_identifier_recursive(child, code)
        if result:
            return result
    return None


#NOT SUMMARIZING CODE BECAUSE OF RATE_LIMIT AND NOT USING LOCAL MODEL BECAUSE OF HARDWARE LIMITATIONS
#def summarize_code(code, fn_name):
    

    #print("========Summarizing Code==========")
    
    #prompt = f'''
     #           You are a helpful code assistant.

      #          Summarize what the following function(given with the function name) does in one or two sentences. 
       #         Focus on the *purpose* of the function, not line-by-line details. 
           #     Avoid repeating variable names unless necessary. 
          #      If the function is a helper or utility, explain what it helps with.

         #       Function {fn_name}:
        #        {code}

 #           '''       
    #summary = models.generate("starcoder2",prompt)
    
    #return summary


def traverse_tree(node, code, path, language_config, context=None):
    if context is None:
        context = []

    chunks = []
    node_category = None

    if node.type in language_config['function_types']:
        node_category = 'function'
    elif node.type in language_config['method_types']:
        node_category = 'method'
    elif node.type in language_config['class_types']:
        node_category = 'class'

    if node_category:
        name = get_identifier_name(node, code, language_config)
        fq_name = '.'.join(context + [name]) if name != "unknown" else name
        chunks.append({
            "path": path,
            "language": language_config['language_name'],
            "category": node_category,
            "node_type": node.type,
            "name": name,
            "fq_name": fq_name,
            "start": node.start_point[0] + 1,
            "end": node.end_point[0] + 1,
            "code": code[node.start_byte:node.end_byte],
            #"summary" : summarize_code(code[node.start_byte:node.end_byte],fq_name)
        })

    # If this is a class, push its name onto the context stack
    new_context = list(context)
    if node.type in language_config['class_types']:
        class_name = get_identifier_name(node, code, language_config)
        if class_name != "unknown":
            new_context.append(class_name)

    # Recurse into children with updated context
    for child in node.children:
        chunks.extend(traverse_tree(child, code, path, language_config, new_context))

    return chunks

def extract_functions_from_file(path, max_lines=1000, parser=None, language_config=None):
    chunks = []
    if parser is None or language_config is None:
        print(f"No parser or language config defined for {path}")
        return chunks
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        
        if len(code.splitlines()) > max_lines:
            print(f"Skipping {path}: too many lines ({len(code.splitlines())} > {max_lines})")
            return chunks
        
        tree = parser.parse(code.encode('utf-8'))
        root_node = tree.root_node
        
        chunks = traverse_tree(root_node, code, path, language_config)
        print(f"Extracted {len(chunks)} items from {path}")
        
    except Exception as e:
        print(f"Could not parse {path}: {e}")
    
    return chunks

def update_index_entry(index_data, file_path, chunks):
    """Update index entry for a file with its parsed chunks and metadata"""
    normalized_path = os.path.normpath(file_path)
    try:
        file_mtime = os.path.getmtime(file_path)
    except OSError:
        file_mtime = 0
    
    index_data[normalized_path] = {
        'last_modified': file_mtime,
        'chunk_count': len(chunks),
        'chunks': chunks,
        'indexed_at':time.time()
    }

def read_files_multi_language(folder, languages=None, index_file='index.json', check_mtime=True):
    """Read files for multiple languages with index-based caching"""
    if languages is None:
        languages = ['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'rust', 'go']
    
    # Load existing index
    index_path = os.path.join(folder, index_file)
    index_data = load_index(index_path)
    
    # Create parsers for each language
    parsers = {}
    for lang in languages:
        try:
            if lang in LanguageConfig.LANGUAGES:
                language = get_language(LanguageConfig.LANGUAGES[lang]['language_name'])
                parser = Parser()
                parser.set_language(language)
                parsers[lang] = parser
                print(f"Loaded parser for {lang}")
        except Exception as e:
            print(f"Could not load parser for {lang}: {e}")
    
    all_chunks = []
    files_processed = 0
    files_skipped = 0
    
    for root, sub_dirs, files in os.walk(folder):
        # Remove common directories to skip
        dirs_to_skip = ['.venv', '__pycache__', '.git', 'node_modules', 'target', 'build', 'dist']
        for skip_dir in dirs_to_skip:
            if skip_dir in sub_dirs:
                sub_dirs.remove(skip_dir)
        
        for file in files:
            # Skip the index file itself
            if file == index_file:
                continue
                
            path = os.path.join(root, file)
            
            # Determine language for this file
            lang_name, lang_config = LanguageConfig.get_language_for_file(path)
            
            if lang_name and lang_name in parsers:
                # Check if we should parse this file
                if should_parse_file(path, index_data, check_mtime):
                    parser = parsers[lang_name]
                    file_chunks = extract_functions_from_file(path, parser=parser, language_config=lang_config)
                    
                    # Update index with new data
                    update_index_entry(index_data, path, file_chunks)
                    all_chunks.extend(file_chunks)
                    files_processed += 1
                else:
                    # Load chunks from index
                    normalized_path = os.path.normpath(path)
                    if normalized_path in index_data and 'chunks' in index_data[normalized_path]:
                        cached_chunks = index_data[normalized_path]['chunks']
                        all_chunks.extend(cached_chunks)
                        files_skipped += 1
    
    # Save updated index
    save_index(index_data, index_path)
    
    print(f"\nProcessed {files_processed} files, skipped {files_skipped} files across {len(parsers)} languages")
    return all_chunks

def print_summary(chunks):
    """Print a nice summary of extracted items"""
    if not chunks:
        print("No functions/classes found!")
        return
    
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Total items: {len(chunks)}")
    

def read_files(folder, languages=None, index_file='index.json', check_mtime=True):
    """Main entry point - supports multiple languages with index-based caching"""
    return read_files_multi_language(folder, languages, index_file, check_mtime)

if __name__ == "__main__":
    folder_path = input("Folder: ")
    
    # chunks = read_files(folder_path, languages=None, index_file='my_index.json', check_mtime=False)
    chunks = read_files(folder_path)
    print_summary(chunks)
