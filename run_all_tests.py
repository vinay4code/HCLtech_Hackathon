import sys
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import unittest
from rag.vector_store import VectorStoreManager
VectorStoreManager.get_client()
VectorStoreManager.get_collection()

suite = unittest.defaultTestLoader.discover("tests")
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
print(f"\nSummary: Ran {result.testsRun} tests. Errors: {len(result.errors)}, Failures: {len(result.failures)}")
if not result.wasSuccessful():
    sys.exit(1)
