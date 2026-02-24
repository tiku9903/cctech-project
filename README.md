# Minimal RAG Chatbot

This is a minimal Retrieval-Augmented Generation (RAG) chatbot example.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Set your OpenAI API key in the environment:

Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

3. Put plain text files into the `docs/` folder (one or more `.txt`).

4. Run the server:

```bash
python run.py
```

5. Send a POST to `/chat` with JSON `{ "question": "..." }`.

Files

- [rag/server.py](rag/server.py): HTTP server and chat endpoint
- [rag/vector_store.py](rag/vector_store.py): simple in-memory vector store using OpenAI embeddings
- [rag/loader.py](rag/loader.py): document loader
- [run.py](run.py): small runner that starts the Flask app

Notes

- This example uses OpenAI embeddings + ChatCompletion; it requires `OPENAI_API_KEY`.
- The vector store is intentionally simple (JSON file) and scales to small corpora for demos.