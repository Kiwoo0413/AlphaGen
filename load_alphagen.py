"""Execute this file from Nuke's Script Editor; no external menu/init edits."""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
import alphagen_nuke

alphagen_nuke.register()
alphagen_nuke.create()
