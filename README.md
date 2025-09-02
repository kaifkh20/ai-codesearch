
# 🚀 AI Code Search

An **AI-powered semantic and keyword code search engine** that helps developers quickly find relevant functions, classes, and snippets across large codebases.

---

## 📖 Overview

AI Code Search is designed to solve the problem of finding specific code without knowing the exact keywords.  
By combining the power of **semantic search** with traditional **lexical methods**, it provides a highly accurate and flexible way to navigate complex projects and monorepos.

---

## ✨ Features (v1.0)

- 🔍 **Semantic Search**: Uses **FAISS** and **embeddings** to understand the intent behind queries, retrieving code snippets even if wording doesn’t match exactly.  
- 📝 **Query Rewriting (Gemini API)**: Automatically expands and clarifies natural language queries using the **Gemini API** for improved results.  
- 📂 **Multi-Language Support**: Parses and indexes code from **Python, C, C++, Go, and Java** with the fast **Tree-sitter** parsing library.  
- 🏷 **Function & Class Extraction**: Indexes functions and classes as discrete units for more precise results compared to file-level searches.  
- 🔑 **Lexical Search**: Falls back to keyword-based matching when semantic results are weak.  
- 🐞 **Basic Bug Search (Python)**: Includes regex-based detection for common Python issues and anti-patterns.  

---

## ⚡ Quick Start

1. Clone the repository and install dependencies.  
   Make sure **Python 3.9** is installed.  

2. Create a `.env` file with:

   ```
   GEMINI_API=YOUR_GEMINI_API_KEY
   ```
3. Run the following:

   ```bash
   git clone https://github.com/kaifkh20/ai-codesearch.git
   cd ai-codesearch
   pip install -r req.txt
   ```

---

## 💡 Usage

To index a codebase and begin searching, run:

```bash
# Example: Search for a function to "calculate the final price with tax"
python main.py -r "folder rep path" -q "function that calculate final price with tax"
```


