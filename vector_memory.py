import chromadb
import requests

CHROMA_DB_PATH = "./chroma_db"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

EMBED_MODEL = "nomic-embed-text"

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_or_create_collection(
    name="neo_memory"
)


def generate_embedding(text):

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "prompt": text
        }
    )

    response.raise_for_status()

    data = response.json()

    return data["embedding"]


def save_memory(text, metadata=None):

    embedding = generate_embedding(text)

    collection.add(
        ids=[str(hash(text))],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata or {}]
    )


def search_memory(query, limit=5):

    embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=limit
    )

    return results