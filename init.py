"""Loaded only when this AlphaGen directory is explicitly added to Nuke."""
import os
import sys
import nuke

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
nuke.pluginAddPath(os.path.join(_root, "gizmos"))
