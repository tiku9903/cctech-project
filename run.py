import os
from rag.server import app, ensure_vector_store
import openai

if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    ensure_vector_store()
    app.run(debug=True, port=7860)
