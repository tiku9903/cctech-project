import os
import openai
from flask import Flask, request, jsonify

from rag.loader import load_documents
from rag.vector_store import SimpleVectorStore, embed_texts


def ensure_vector_store(doc_dir="docs", store_path="vector_store.json"):
    if not os.path.exists(store_path):
        docs = load_documents(doc_dir)
        vs = SimpleVectorStore(path=store_path)
        if docs:
            texts = [d["text"] for d in docs]
            embs = embed_texts(texts)
            vs.add(docs, embs)
        else:
            # ensure file exists
            vs._save()


app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True)
    question = payload.get("question")
    k = int(payload.get("k", 3))
    if not question:
        return jsonify({"error": "missing question"}), 400

    vs = SimpleVectorStore()
    hits = vs.query(question, k=k)
    context = "\n\n---\n\n".join([h.get("text", "") for h in hits])

    prompt = (
        "You are a helpful assistant. Use the provided context to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    resp = openai.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
    )
    answer = resp["choices"][0]["message"]["content"]
    return jsonify({"answer": answer, "sources": [h.get("id") for h in hits]})


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    ensure_vector_store()
    app.run(host="0.0.0.0", port=7860)
