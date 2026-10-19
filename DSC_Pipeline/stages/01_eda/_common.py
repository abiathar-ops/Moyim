# Thin shim: ensure the pipeline root (dir containing src/) is importable,
# then hand off to src/_common for the real logic.
import sys
from pathlib import Path
_here = Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "src").is_dir():
        sys.path.insert(0, str(_p)); break
from src import _common  # noqa: F401,E402
