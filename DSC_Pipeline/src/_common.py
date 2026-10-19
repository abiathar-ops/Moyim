"""Path bootstrap for stage scripts, wherever they live in the tree.

Walks up from this file to the pipeline root (the directory that contains
``src``) and puts it on sys.path, so ``from src import config`` works whether
a script sits in ``stages/03_modeling/`` or anywhere else.
"""
import sys
from pathlib import Path

def _pipeline_root():
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "src").is_dir():
            return p
    return here.parents[1]

ROOT = _pipeline_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
