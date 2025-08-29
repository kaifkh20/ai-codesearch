import os
import ast

def extract_functions_from_file(path):
    chunks = []

    with open(path,"r") as f:
        code = f.read()
    tree = ast.parse(code)
    try:
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef):
                start = node.lineno
                end = max(
                    [n.lineno for n in ast.walk(node) if hasattr(n,"lineno")],
                    default = start #ensure that if there is no function def then first line is given
                )
                lines = code.splitlines()[start-1:end] #breaks function in lines
                chunks.append((node.name,start,end,"\n".join(lines)))
    except Exception as e:
        print(f"Couldn't parse path {path} : {e}")
    return chunks

def read_files_python(folder):
    file_chunks = []
    for root,sub_dirs,files in os.walk(folder):
        if ".venv" in sub_dirs:
            #continue doesnt work here because os.walk() iterator has already planned on exploring the .venv at this point
            sub_dirs.remove(".venv")
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root,file)
                chunks_functions = extract_functions_from_file(path)
                if chunks_functions:
                    for fn_name,start,end,code in chunks_functions:
                        file_chunks.append((path,fn_name,start,end,code))               
                
                #fallback if file doesnt have functions               
                else:
                    try:
                        with open(path,"r",encoding="utf-8") as f:
                            text = f.read()
                            lines = len(text.splitlines())
                        file_chunks.append((path,"<file>",1,lines,text))

                    except Exception as e:
                        print(f"Could not read {path} : {e}")
    return file_chunks

def read_files(folder):
    return read_files_python(folder)

def search_response(chunks,query):
    results = []
    for path,fn_name,start,end,code in chunks:
        if query.lower() in code.lower():
            results.append((path,fn_name,start,end))
    return results

def format_response(results):
    if results:
        print("Found in:")
        for path, fn_name, start, end in results:
            print(f" - {path}:{start}-{end}  (function: {fn_name})")
    else:
        print("No matches found.")

def search(folder,query):
    code_chunks = read_files_python(folder)
    result = search_response(code_chunks,query)

    format_response(result)

