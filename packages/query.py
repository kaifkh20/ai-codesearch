import os

def read_files_python(folder):
    code_chunks = []
    for root,sub_dirs,files in os.walk(folder):
        if ".venv" in sub_dirs:
            #continue doesnt work here because os.walk() iterator has already planned on exploring the .venv at this point
            sub_dirs.remove(".venv")
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root,file)

                try:
                    with open(path,"r",encoding="utf-8") as f:
                        lines = f.readlines()
                        code_chunks.append((path,lines))

                except Exception as e:
                    print(f"Could not read {path} : {e}")
    return code_chunks

def read_files(folder):
    return read_files_python(folder)

def search_response(chunks,query):
    results = []
    for path,lines in chunks:
        for i,line in enumerate(lines,1):
            if query.lower() in line.lower():
                results.append((path,i,line.strip()))
    return results

def format_response(results):
        if results:
            print("Found in:")
            for (path,line_no,line_text) in results:
                print(f"- {path}:{line_no} -> {line_text}")
        else:
            print("No matches found.")

def search(folder,query):
    code_chunks = read_files_python(folder)
    result = search_response(code_chunks,query)

    format_response(result)

