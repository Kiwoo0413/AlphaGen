"""Parse the serialized Tcl stack and check wiring; this is not a Nuke render test."""
import json
from pathlib import Path
import re
import tkinter

ROOT = Path(__file__).resolve().parents[1]
interpreter = tkinter.Tcl()
stack = []
nodes = {}


def node_command(kind):
    def command(body):
        name = re.search(r"^\s*name\s+(\S+)", body, re.M).group(1)
        match = re.search(r"^\s*inputs\s+(\d+)", body, re.M)
        count = int(match.group(1)) if match else 1
        inputs = [stack.pop() for _ in range(count)]
        nodes[name] = {"class": kind, "inputs": inputs}
        stack.append(name)
        return name
    return command


for kind in ("Input", "Constant", "Read", "Expression", "Merge2", "Copy", "Switch", "Premult", "Output"):
    interpreter.createcommand(kind, node_command(kind))
interpreter.createcommand("Gizmo", lambda body: "")
interpreter.createcommand("end_group", lambda: "")
interpreter.createcommand("stack", lambda index: stack[-1 - int(index)])
interpreter.createcommand("push", lambda name: stack.append(name) or "")
interpreter.eval((ROOT / "gizmos/AlphaGen.gizmo").read_text(encoding="utf-8"))
expected = {
    "OptionalCore": ["EmptyMask", "CoreAdd"],
    "OptionalGarbage": ["EmptyMask", "Garbage"],
    "SelectSource": ["VlmAlpha", "Selection"],
    "AddCore": ["RefinedAlpha", "OptionalCore"],
    "RemoveGarbage": ["AddCore", "OptionalGarbage"],
    "SourceWithAlpha": ["Source", "FinalAlpha"],
    "Display": ["FinalAlpha", "RawAlpha", "SelectionAlpha", "CoreAlpha", "EdgeAlpha", "SourceWithAlpha", "Premultiplied"],
    "ReadyGate": ["Source", "Display"],
    "Output": ["ReadyGate"],
}
for name, inputs in expected.items():
    assert nodes[name]["inputs"] == inputs, (name, nodes[name])
print("Gizmo Tcl syntax and 8 critical node connections passed (not a Nuke render test)")

