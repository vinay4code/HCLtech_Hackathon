import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Initialize ChromaDB native runtime before any OpenMP/scikit-learn runtime initialization
try:
    from rag.vector_store import VectorStoreManager
    VectorStoreManager.get_client()
except Exception:
    pass
