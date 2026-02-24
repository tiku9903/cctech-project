import os

def load_documents(doc_dir="docs"):
    """Load all text files from `doc_dir` into a list of dicts {id, text}."""
    docs = []
    if not os.path.exists(doc_dir):
        return docs
    for fname in sorted(os.listdir(doc_dir)):
        path = os.path.join(doc_dir, fname)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                docs.append({"id": fname, "text": text})
            except Exception:
                continue
    return docs
