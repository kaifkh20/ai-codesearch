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

MAX_LINES_PER_CHUNK = 300

def split_large_code_block(code, max_lines=MAX_LINES_PER_CHUNK):
    lines = code.splitlines()
    for i in range(0, len(lines), max_lines):
        yield "\n".join(lines[i:i+max_lines])

class LanguageConfig:
    """Configuration for different programming languages"""
    
    LANGUAGES = {
        'python': {
            'extensions': ['.py'],
            'language_name': 'python',
            'function_types': ['function_definition'],
            'class_types': ['class_definition'],
            'method_types': ['function_definition'],
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
            'function_types': ['function_definition'],
            'class_types': ['class_specifier'],
            'method_types': ['function_definition'],
        },
        'c': {
            'extensions': ['.c', '.h'],
            'language_name': 'c',
            'function_types': ['function_definition'],
            'class_types': [],
            'method_types': [],
        },
        'rust': {
            'extensions': ['.rs'],
            'language_name': 'rust',
            'function_types': ['function_item'],
            'class_types': ['struct_item', 'enum_item', 'impl_item'],
            'method_types': ['function_item'],
        },
        'go': {
            'extensions': ['.go'],
            'language_name': 'go',
            'function_types': ['function_declaration', 'method_declaration'],
            'class_types': ['type_declaration'],
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
    """Check if a file should be parsed based on index data."""
    normalized_path = os.path.normpath(file_path)
    
    if normalized_path not in index_data:
        return True
    
    if not check_mtime:
        print(f"Skipping {file_path}: found in index")
        return False
    
    try:
        file_mtime = os.path.getmtime(file_path)
        indexed_mtime = index_data[normalized_path].get('last_modified', 0)
        
        if file_mtime > indexed_mtime:
            print(f"Re-parsing {file_path}: file modified since last index")
            return True
        else:
            print(f"Skipping {file_path}: up to date in index")
            return False
            
    except (OSError, KeyError, TypeError):
        print(f"Re-parsing {file_path}: could not verify modification time")
        return True

def clean_identifier_name(name):
    """Clean and sanitize identifier names"""
    if not name or name == "unknown":
        return "unknown"
    
    # Remove common prefixes/suffixes that might be noise
    name = name.strip()
    
    # Remove quotes if present
    if name.startswith('"') and name.endswith('"'):
        name = name[1:-1]
    if name.startswith("'") and name.endswith("'"):
        name = name[1:-1]
    
    # Remove whitespace and newlines
    name = ''.join(name.split())
    
    # If name is empty after cleaning, return unknown
    if not name:
        return "unknown"
    
    return name

def get_identifier_name(node, code, language_name=None):
    """Extract identifier name from AST node with improved cleaning"""
    
    # Direct field name (works for many languages)
    name_node = node.child_by_field_name('name')
    if name_node:
        name = code[name_node.start_byte:name_node.end_byte]
        return clean_identifier_name(name)
    
    # Language-specific handling
    if language_name == 'c' or language_name == 'cpp':
        name = get_c_cpp_identifier(node, code)
    elif language_name == 'java':
        name = get_java_identifier(node, code)
    elif language_name == 'rust':
        name = get_rust_identifier(node, code)
    elif language_name == 'go':
        name = get_go_identifier(node, code)
    elif language_name in ['javascript', 'typescript']:
        name = get_js_ts_identifier(node, code)
    else:
        # Fallback: look for any identifier in immediate children
        name = None
        for child in node.children:
            if child.type in ('identifier', 'property_identifier', 'type_identifier', 'field_identifier'):
                name = code[child.start_byte:child.end_byte]
                break
    
    return clean_identifier_name(name) if name else "unknown"

def get_c_cpp_identifier(node, code):
    """Extract identifier for C/C++ functions with better error handling"""
    if node.type == "function_definition":
        declarator = node.child_by_field_name("declarator")
        if declarator:
            name = _find_identifier_recursive(declarator, code)
            if name:
                return name
        
        # Look for function_declarator
        for child in node.children:
            if child.type == "function_declarator":
                inner_declarator = child.child_by_field_name("declarator")
                if inner_declarator and inner_declarator.type == "identifier":
                    return code[inner_declarator.start_byte:inner_declarator.end_byte]
                
                # Look for first identifier
                for grandchild in child.children:
                    if grandchild.type == "identifier":
                        return code[grandchild.start_byte:grandchild.end_byte]
        
        # Last resort
        return _find_identifier_recursive(node, code)
    
    elif node.type in ["class_specifier", "struct_specifier"]:
        name_node = node.child_by_field_name('name')
        if name_node:
            return code[name_node.start_byte:name_node.end_byte]
        
        # Look for identifier after class/struct keyword
        for i, child in enumerate(node.children):
            if child.type in ["class", "struct"] and i + 1 < len(node.children):
                next_child = node.children[i + 1]
                if next_child.type == "type_identifier":
                    return code[next_child.start_byte:next_child.end_byte]
    
    return _find_identifier_recursive(node, code)

def get_java_identifier(node, code):
    """Extract identifier for Java methods/classes"""
    name_node = node.child_by_field_name('name')
    if name_node:
        return code[name_node.start_byte:name_node.end_byte]
    
    if node.type == "method_declaration":
        for child in node.children:
            if child.type == "identifier":
                return code[child.start_byte:child.end_byte]
    
    return _find_identifier_recursive(node, code)

def get_rust_identifier(node, code):
    """Extract identifier for Rust functions/structs/enums"""
    name_node = node.child_by_field_name('name')
    if name_node:
        return code[name_node.start_byte:name_node.end_byte]
    
    for i, child in enumerate(node.children):
        if child.type in ["fn", "struct", "enum", "impl"] and i + 1 < len(node.children):
            next_child = node.children[i + 1]
            if next_child.type in ["identifier", "type_identifier"]:
                return code[next_child.start_byte:next_child.end_byte]
    
    return _find_identifier_recursive(node, code)

def get_go_identifier(node, code):
    """Extract identifier for Go functions"""
    name_node = node.child_by_field_name('name')
    if name_node:
        return code[name_node.start_byte:name_node.end_byte]
    
    if node.type in ["function_declaration", "method_declaration"]:
        for i, child in enumerate(node.children):
            if child.type == "func" and i + 1 < len(node.children):
                next_child = node.children[i + 1]
                if next_child.type == "identifier":
                    return code[next_child.start_byte:next_child.end_byte]
    
    return _find_identifier_recursive(node, code)

def get_js_ts_identifier(node, code):
    """Extract identifier for JavaScript/TypeScript functions with better anonymous handling"""
    name_node = node.child_by_field_name('name')
    if name_node:
        return code[name_node.start_byte:name_node.end_byte]
    
    # Handle different function types more specifically
    if node.type == "arrow_function":
        # Try to find if it's assigned to a variable
        parent = node.parent
        if parent and parent.type == "variable_declarator":
            name_node = parent.child_by_field_name('name')
            if name_node:
                return f"arrow_{code[name_node.start_byte:name_node.end_byte]}"
        return "anonymous_arrow"
    
    elif node.type == "function_expression":
        # Try to find if it's assigned or a property
        parent = node.parent
        if parent:
            if parent.type == "variable_declarator":
                name_node = parent.child_by_field_name('name')
                if name_node:
                    return f"expr_{code[name_node.start_byte:name_node.end_byte]}"
            elif parent.type == "pair":  # Object property
                key_node = parent.child_by_field_name('key')
                if key_node:
                    return code[key_node.start_byte:key_node.end_byte]
        return "anonymous_function"
    
    elif node.type == "method_definition":
        key_node = node.child_by_field_name('key')
        if key_node:
            return code[key_node.start_byte:key_node.end_byte]
    
    return _find_identifier_recursive(node, code)

def _find_identifier_recursive(node, code):
    """Recursively search for an identifier with better filtering"""
    if not node:
        return None
        
    if node.type in ["identifier", "type_identifier", "field_identifier"]:
        return code[node.start_byte:node.end_byte]
    
    for child in node.children:
        result = _find_identifier_recursive(child, code)
        if result and result.strip():
            return result
    return None

def traverse_tree(node, code, path, language_config, context=None):
    if context is None:
        context = []

    chunks = []
    node_category = None
    language_name = language_config['language_name']

    # Determine node category
    if node.type in language_config['function_types']:
        if language_name in ['c', 'cpp'] and node.type == 'function_declarator':
            pass
        else:
            node_category = 'function'
    elif node.type in language_config['method_types']:
        node_category = 'method'
    elif node.type in language_config['class_types']:
        node_category = 'class'

    if node_category:
        name = get_identifier_name(node, code, language_name)
        
        # Skip if we couldn't extract a meaningful name
        if not name or name == "unknown":
            print(f"Warning: Could not extract name for {node_category} at line {node.start_point[0] + 1} in {path}")
            name = f"unnamed_{node_category}_{node.start_point[0] + 1}"
        
        fq_name = '.'.join(context + [name]) if context else name
        function_code = code[node.start_byte:node.end_byte]

        # Chunking with improved naming
        for i, sub_code in enumerate(split_large_code_block(function_code)):
            chunk_name = f"{name}_part{i+1}" if i > 0 else name
            chunks.append({
                "path": path,
                "language": language_name,
                "category": node_category,
                "node_type": node.type,
                "name": chunk_name,
                "fq_name": fq_name,
                "start": node.start_point[0] + 1,
                "end": node.end_point[0] + 1,
                "code": sub_code,
            })

    # Update context for nested structures
    new_context = list(context)
    if node.type in language_config['class_types']:
        class_name = get_identifier_name(node, code, language_name)
        if class_name and class_name != "unknown":
            new_context.append(class_name)

    if language_name == 'rust' and node.type == 'impl_item':
        type_node = node.child_by_field_name('type')
        if type_node:
            impl_type = code[type_node.start_byte:type_node.end_byte]
            impl_type = clean_identifier_name(impl_type)
            if impl_type != "unknown":
                new_context.append(f"impl_{impl_type}")

    # Recursively process children
    for child in node.children:
        chunks.extend(traverse_tree(child, code, path, language_config, new_context))

    return chunks

def extract_functions_from_file(path, max_lines=1000, parser=None, language_config=None, max_body_preview=50):
    """Extract functions/classes from file using tree-sitter with better error handling"""
    chunks = []
    if parser is None or language_config is None:
        print(f"No parser or language config defined for {path}")
        return chunks
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        
        tree = parser.parse(code.encode("utf-8"))
        root_node = tree.root_node
        
        chunks = traverse_tree(root_node, code, path, language_config)

        # Truncate huge chunks safely
        safe_chunks = []
        for chunk in chunks:
            code_lines = chunk["code"].splitlines()
            if len(code_lines) > max_lines:
                signature = code_lines[0] if code_lines else ""
                docstring = ""
                if len(code_lines) > 1 and code_lines[1].strip().startswith(('"""', "'''", "/*", "//")):
                    docstring = code_lines[1]
                body_preview = "\n".join(code_lines[1:1 + max_body_preview])
                truncated = "\n".join(filter(None, [
                    signature,
                    docstring,
                    body_preview,
                    "    # ... [truncated] ..."
                ]))
                chunk["code"] = truncated
            safe_chunks.append(chunk)
        
        print(f"Extracted {len(safe_chunks)} items from {path}")
        return safe_chunks
    
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
        'indexed_at': time.time()
    }

def read_files_multi_language(folder, languages=None, index_file='index.json', check_mtime=True):
    """Read files for multiple languages with index-based caching"""
    if languages is None:
        languages = ['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'rust', 'go']
    
    if not os.path.exists(folder):
        print("NO SUCH REPO EXISTS")
        sys.exit(1)
    
    index_path = "index.json"
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
        # Comprehensive list of directories to skip
        dirs_to_skip = [
            # Documentation
            'docs', 'docs_src', 'documentation', 'doc', 'man', 'manual',
            
            # Testing
            'tests', 'test', 'testing', '__tests__', 'spec', 'specs', 'e2e', 'integration', 'unit',
            
            # Build/Output directories
            'build', 'dist', 'out', 'output', 'target', 'bin', 'obj', 'release', 'debug',
            'public', 'static', 'assets', 'generated', 'gen', 'tmp', 'temp', 'cache',
            
            # Dependencies/Libraries
            'node_modules', 'bower_components', 'vendor', 'vendors', 'lib', 'libs', 
            'third_party', 'external', 'deps',
            
            # Python specific
            '.venv', 'venv', 'env', '__pycache__', '.pytest_cache', '.tox', 
            'site-packages', '.mypy_cache', '.coverage', 'htmlcov',
            
            # Version control
            '.git', '.svn', '.hg', '.bzr', 'CVS',
            
            # IDE/Editor directories
            '.vscode', '.idea', '.eclipse', '.settings', '.project', '.metadata',
            '.vs', '.suo', '.user', 'nbproject',
            
            # Language/Framework specific build dirs
            'target', 'classes', '.gradle', 'gradle',
            'bin', 'obj', 'packages', '.nuget',
            '.bundle', 'vendor/bundle',
            'vendor', 'composer',
            'coverage', '.nyc_output', '.next', '.nuxt', '.parcel-cache',
            'ios/build', 'android/build', '.expo', '.expo-shared',
            
            # CI/CD and deployment
            '.github', '.gitlab', '.circleci', '.travis', '.appveyor', 
            'deployment', 'deploy', '.aws', '.terraform',
            
            # Logs and runtime
            'logs', 'log', '.log', 'crash-reports', 'error-reports',
            
            # OS specific
            '.DS_Store', 'Thumbs.db', 'desktop.ini',
            
            # Package managers
            '.npm', '.yarn', '.pnpm-store', '.composer', '.pip',
            
            # Backup and temporary
            'backup', 'backups', '.backup', '.bak', '.tmp', '.temp',
            
            # Media and assets
            'images', 'img', 'pictures', 'videos', 'audio', 'fonts', 'media',
            'resources', 'res', 'assets/img', 'assets/images',
            
            # Localization
            'locale', 'locales', 'i18n', 'l10n', 'translations',
            
            # Migrations and seeds
            'migrations', 'seeds', 'fixtures',
            
            # Configuration directories
            'config', 'conf', 'cfg', '.config', 'settings',
            
            # Examples and samples
            'examples', 'example', 'samples', 'sample', 'demo', 'demos',
            
            # Generated code directories
            'generated', 'gen', 'autogen', 'codegen', 'proto', 'protobuf',
            
            # Docker
            '.docker', 'docker',
            
            # Jupyter/IPython
            '.ipynb_checkpoints',
            
            # R
            '.Rproj.user',
            
            # Elixir
            '_build', 'deps',
            
            # Flutter
            '.dart_tool', '.flutter-plugins', '.flutter-plugins-dependencies',
            
            # Unity
            'Library', 'Temp', 'Logs',
            
            # Xcode
            '*.xcworkspace', '*.xcodeproj', 'DerivedData',
            
            # Android
            '.gradle', 'build', 'captures', '.externalNativeBuild',
            
            # Web specific
            'bower_components', 'jspm_packages', 'web_modules',
        ]
        
        skip_patterns = [
            'test_', 'tests_', '_test', '_tests',
            'spec_', 'specs_', '_spec', '_specs',
            'build_', '_build', 'dist_', '_dist',
            'temp_', '_temp', 'tmp_', '_tmp',
            'cache_', '_cache', 'backup_', '_backup',
        ]
        
        # Remove directories to skip
        for skip_dir in dirs_to_skip:
            if skip_dir in sub_dirs:
                sub_dirs.remove(skip_dir)
        
        # Remove directories matching patterns
        dirs_to_remove = []
        for dir_name in sub_dirs:
            dir_lower = dir_name.lower()
            for pattern in skip_patterns:
                if pattern in dir_lower:
                    dirs_to_remove.append(dir_name)
                    break
        
        for dir_name in dirs_to_remove:
            sub_dirs.remove(dir_name)
        
        for file in files:
            if file == index_file:
                continue
                
            path = os.path.join(root, file)
            
            lang_name, lang_config = LanguageConfig.get_language_for_file(path)
            
            if lang_name and lang_name in parsers:
                if should_parse_file(path, index_data, check_mtime):
                    parser = parsers[lang_name]
                    file_chunks = extract_functions_from_file(path, parser=parser, language_config=lang_config)
                    
                    update_index_entry(index_data, path, file_chunks)
                    all_chunks.extend(file_chunks)
                    files_processed += 1
                else:
                    normalized_path = os.path.normpath(path)
                    if normalized_path in index_data and 'chunks' in index_data[normalized_path]:
                        cached_chunks = index_data[normalized_path]['chunks']
                        all_chunks.extend(cached_chunks)
                        files_skipped += 1

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
    
    by_language = {}
    for chunk in chunks:
        lang = chunk['language']
        category = chunk['category']
        if lang not in by_language:
            by_language[lang] = {}
        if category not in by_language[lang]:
            by_language[lang][category] = 0
        by_language[lang][category] += 1
    
    for lang, categories in by_language.items():
        print(f"\n{lang.upper()}:")
        for category, count in categories.items():
            print(f"  {category}s: {count}")
    
    print(f"\nSample extracted items:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"{i+1}. [{chunk['language']}] {chunk['category']} '{chunk['name']}' in {os.path.basename(chunk['path'])}")

def read_files(folder, languages=None, index_file='index.json', check_mtime=True):
    """Main entry point - supports multiple languages with index-based caching"""
    return read_files_multi_language(folder, languages, index_file, check_mtime)

if __name__ == "__main__":
    folder_path = input("Folder: ")
    
    chunks = read_files(folder_path)
    print_summary(chunks)
