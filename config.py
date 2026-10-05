import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "20"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
NO_INFO_PHRASE = "I couldn't find that information in the uploaded documents."
