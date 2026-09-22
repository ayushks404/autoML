"""
RAG memory over past training runs, using FAISS + sentence-transformers.

The embedding model is downloaded from HuggingFace on first use, so this
module lazily initializes only when store_experiment/retrieve_similar are
actually called, and every public function catches its own exceptions --
if there's no network access or the packages aren't installed, training
continues without memory instead of crashing.
"""
import os
import pickle

import numpy as np

INDEX_FILE = os.environ.get("RAG_INDEX_FILE", "rag_index.faiss")
DATA_FILE = os.environ.get("RAG_DATA_FILE", "rag_data.pkl")
EMBEDDING_DIM = 384

_model = None
_index = None
_stored_data = None


def _lazy_init():
    global _model, _index, _stored_data
    if _model is not None:
        return
    import faiss
    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer("all-MiniLM-L6-v2")
    _index = faiss.read_index(INDEX_FILE) if os.path.exists(INDEX_FILE) else faiss.IndexFlatL2(EMBEDDING_DIM)

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            _stored_data = pickle.load(f)
    else:
        _stored_data = []


def store_experiment(text: str) -> bool:
    try:
        import faiss  # noqa: F401 (re-imported here so the write below can reference it)

        _lazy_init()
        embedding = _model.encode([text])
        _index.add(np.array(embedding, dtype="float32"))
        _stored_data.append(text)

        import faiss as _faiss
        _faiss.write_index(_index, INDEX_FILE)
        with open(DATA_FILE, "wb") as f:
            pickle.dump(_stored_data, f)
        return True
    except Exception as e:
        print("⚠️ RAG store skipped (continuing without memory):", e)
        return False


def retrieve_similar(query: str, k: int = 3):
    try:
        _lazy_init()
        if not _stored_data:
            return []
        query_embedding = _model.encode([query])
        distances, indices = _index.search(np.array(query_embedding, dtype="float32"), min(k, len(_stored_data)))
        results = [_stored_data[i] for i in indices[0] if 0 <= i < len(_stored_data)]
        return list(dict.fromkeys(results))[:k]  # de-dupe, preserve order
    except Exception as e:
        print("⚠️ RAG retrieve skipped (continuing without memory):", e)
        return []
