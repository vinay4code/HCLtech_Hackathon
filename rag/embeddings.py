import os
import numpy as np
from typing import List, Union
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from config.settings import LOCAL_HF_MODEL_PATH, EMBEDDING_PROVIDER, GEMINI_API_KEY

class RobustEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function supporting Local HuggingFace SentenceTransformer,
    Scikit-Learn Hashing/TF-IDF Vectorizer (for zero-latency offline runs),
    and Gemini REST API embeddings.
    """
    def __init__(self, provider: str = None):
        self.provider = provider or EMBEDDING_PROVIDER
        self._hf_model = None
        self._hasher = None
        self.dimension = 384

    def _get_hf_model(self):
        if self._hf_model is None:
            from sentence_transformers import SentenceTransformer
            if os.path.exists(LOCAL_HF_MODEL_PATH):
                self._hf_model = SentenceTransformer(LOCAL_HF_MODEL_PATH, local_files_only=True)
            else:
                self._hf_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._hf_model

    def _get_hasher(self):
        if self._hasher is None:
            from sklearn.feature_extraction.text import HashingVectorizer
            self._hasher = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2")
        return self._hasher

    def __call__(self, input: Documents) -> Embeddings:
        docs = list(input)
        if not docs:
            return []

        # If provider is LOCAL_HF and model path exists
        if self.provider == "LOCAL_HF":
            try:
                model = self._get_hf_model()
                vectors = model.encode(docs, convert_to_numpy=True)
                return vectors.tolist()
            except Exception as e:
                # Fallback to deterministic Hashing vectorizer
                pass

        # If Gemini API requested and key present
        if self.provider == "GEMINI" and GEMINI_API_KEY:
            try:
                import requests
                url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={GEMINI_API_KEY}"
                requests_body = {
                    "requests": [{"model": "models/text-embedding-004", "content": {"parts": [{"text": d}]}} for d in docs]
                }
                res = requests.post(url, json=requests_body, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    return [e["values"][:384] for e in data.get("embeddings", [])]
            except Exception:
                pass

        # Fast deterministic fallback: HashingVectorizer (L2 normalized 384-dim)
        hasher = self._get_hasher()
        sparse_matrix = hasher.transform(docs)
        dense_matrix = sparse_matrix.toarray().astype(np.float32)
        return dense_matrix.tolist()

    @property
    def model_name(self) -> str:
        if self.provider == "LOCAL_HF":
            return "all-MiniLM-L6-v2 (Local PyTorch)"
        elif self.provider == "GEMINI":
            return "text-embedding-004 (Gemini API)"
        return "HashingVectorizer-384 (Fast Normalized)"

    @staticmethod
    def name() -> str:
        return "RobustEmbeddingFunction"


    def get_config(self) -> dict:
        return {"provider": self.provider}

    @classmethod
    def build_from_config(cls, config: dict) -> "RobustEmbeddingFunction":
        return cls(provider=config.get("provider"))



