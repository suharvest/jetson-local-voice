import os
import sys

# Make project root (parent of app/) importable so `from app.core.* import`
# resolves the same way as in production (uvicorn app.main:app).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
