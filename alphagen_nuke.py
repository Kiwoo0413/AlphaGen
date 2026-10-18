"""AlphaGen Nuke UI bridge.

Manages Nuke GUI knobs, real-time background progress monitoring,
VLM spatial grounding, and live preview of generated mattes on a unified single tab.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import uuid

try:
    from PySide2 import QtCore
except ImportError:
    try:
        from PySide6 import QtCore
    except ImportError:
        QtCore = None

ROOT = Path(__file__).resolve().parent
ACTIVE_TIMERS: dict[str, QtCore.QTimer] = {}
ACTIVE_PROCESSES: dict[str, subprocess.Popen] = {}

SETTING_DEFAULTS = {
    "black_clip": 0.05, "white_clip": 0.95, "gamma": 1.0,
    "core_threshold": 0.6, "core_inset": 3, "core_strength": 1.0,
    "fill_holes": False, "max_hole_area": 64, "safe_radius": 6,
    "edge_restore": 0.5, "feather": 0.35,
}


def _nuke():
    import nuke
    return nuke


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _settings(node):
    return {key: node[key].value() for key in SETTING_DEFAULTS if key in node.knobs()}


def resolve_python():
    env_py = os.environ.get("ALPHAGEN_PYTHON")
    if env_py and Path(env_py).is_file():
        return Path(env_py)
    candidates = [
        Path("D:/AI/ComfyUI_DATA/.venv/Scripts/python.exe"),
        ROOT / ".venv/Scripts/python.exe",
        ROOT / ".venv/bin/python",
    ]
    which_py = shutil.which("python")
    if which_py and "nuke" not in which_py.lower():
        candidates.append(Path(which_py))
    for c in candidates:
        if c.is_file() and "nuke" not in str(c).lower():
            return c
    if "nuke" not in sys.executable.lower():
        return Path(sys.executable)
    raise RuntimeError(
        "AlphaGen AI Python environment not found!\n"
        "Please specify your Python executable in ~/.nuke/init.py:\n"
        "os.environ['ALPHAGEN_PYTHON'] = r'C:/path/to/python.exe'\n"
        "or install AlphaGen requirements into your environment:\n"
        "pip install -r requirements.txt"
    )


def create(show_panel=True):
    nuke = _nuke()
    nuke.pluginAddPath(str(ROOT / "gizmos"))
    node = nuke.createNode("AlphaGen", inpanel=False)
    configure_graph(node)
    node["first"].setValue(int(nuke.frame()))
    node["last"].setValue(int(nuke.frame()) + 16)
    if show_panel:
        node.showControlPanel()
    return node


def configure_graph(node):
    """Apply fixes and configure internal gizmo nodes."""
    node["onCreate"].setValue("import alphagen_nuke; alphagen_nuke.restore_saved(nuke.thisNode())")
    if "cancel" in node.knobs():
        node["cancel"].setLabel("<b><font color='#FF5252'>⏹ Stop Queue (중단)</font></b>")
    for name in ("AddCore", "RemoveGarbage"):
        merge = node.node(name)
        if merge:
            merge["Achannels"].setValue("rgba")
            merge["Bchannels"].setValue("rgba")
            merge["output"].setValue("rgba")
            merge["screen_alpha"].setValue(False)
            merge["bbox"].setValue("B")


def restore_saved(node):
    """Restore cache paths on gizmo load or script reopen."""
    if "result_path" in node.knobs():
        path = node["result_path"].value()
        if path and Path(path).is_file():
            try:
                load_result(node, path)
            except Exception as exc:
                set_status(node, "Restore failed: " + str(exc))
    if "vlm_mask_path" in node.knobs():
        vlm_path = node["vlm_mask_path"].value()
        if vlm_path and Path(vlm_path).is_file():
            load_vlm_preview(node, vlm_path)


def format_signature(node):
    f = node.format()
    return (f.width(), f.height(), f.x(), f.y(), f.r(), f.t(), f.pixelAspect())


def get_upstream_range(upstream):
    """Accurately extract start and end frame numbers from upstream node tree."""
    if upstream:
        # 1. Try Read node knobs directly
        if hasattr(upstream, "knobs") and "first" in upstream.knobs() and "last" in upstream.knobs():
            try:
                f = int(upstream["first"].value())
                l = int(upstream["last"].value())
                if l >= f:
                    return f, l
            except Exception:
                pass

        # 2. Try standard node methods
        try:
            f = int(upstream.firstFrame())
            l = int(upstream.lastFrame())
            if l >= f:
                return f, l
        except Exception:
            pass

    try:
        nuke = _nuke()
        return int(nuke.root().firstFrame()), int(nuke.root().lastFrame())
    except Exception:
        return 1, 100


def sync_source_range(node):
    """Automatically detect and set the valid 8n+1 frame range from Source input."""
    source = node.input(0)
    if not source:
        return
    f, l = get_upstream_range(source)
    total = max(1, l - f + 1)
    # LTX requires temporal factor 8n + 1 (unconstrained via 49-frame chunking pipeline)
    n_steps = max(1, (total - 1) // 8)
    valid_count = n_steps * 8 + 1
    valid_last = f + valid_count - 1

    if "first" in node.knobs():
        node["first"].setValue(f)
    if "last" in node.knobs():
        node["last"].setValue(valid_last)
    if "ag_ref_frame" in node.knobs():
        node["ag_ref_frame"].setValue(f)

    set_status(node, f"Frame Range auto-set to {f}–{valid_last} ({valid_count} frames, 8n+1)")


def get_source_file_path(upstream):
    """Trace upstream node tree to locate the original source media file path."""
    if not upstream:
        return None
    visited = set()
    stack = [upstream]
    while stack:
        curr = stack.pop(0)
        if not curr or curr in visited:
            continue
        visited.add(curr)
        if "file" in curr.knobs():
            val = curr["file"].value()
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        for i in range(curr.inputs()):
            inp = curr.input(i)
            if inp and inp not in visited:
                stack.append(inp)
    return None


def get_source_alpha_dir(node):
    """Get or create the 'alpha' folder in the directory of the source video."""
    source_node = node.input(0)
    file_path = get_source_file_path(source_node)
    if not file_path:
        return None
    try:
        clean = file_path.replace("\\", "/")
        source_dir = os.path.dirname(clean)
        if source_dir:
            alpha_dir = Path(source_dir) / "alpha"
            alpha_dir.mkdir(parents=True, exist_ok=True)
            return alpha_dir.as_posix()
    except Exception:
        pass
    return None


def get_nuke_cache_dir() -> Path:
    """Returns the cache directory based on Nuke's preferences and environment.
    Priority:
    1. ALPHAGEN_CACHE_DIR environment variable (manual user override)
    2. Nuke Preferences 'localCachePath' (evaluated)
    3. Nuke Preferences 'DiskCachePath' (evaluated)
    4. NUKE_TEMP_DIR environment variable
    5. OS temp / nuke / AlphaGen
    6. Fallback: ROOT / 'jobs'
    """
    override = os.environ.get("ALPHAGEN_CACHE_DIR")
    if override and override.strip():
        p = Path(override.strip()).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    try:
        nuke = _nuke()
        pref = nuke.toNode("preferences")
        if pref:
            for knob_name in ("localCachePath", "DiskCachePath"):
                if knob_name in pref.knobs():
                    val = None
                    try:
                        val = pref[knob_name].evaluate()
                    except Exception:
                        val = pref[knob_name].value()
                    if val and isinstance(val, str) and val.strip():
                        val_str = val.strip()
                        if not val_str.startswith("[") and not val_str.startswith("$"):
                            p = Path(val_str).resolve() / "AlphaGen"
                            p.mkdir(parents=True, exist_ok=True)
                            return p
    except Exception:
        pass

    env_temp = os.environ.get("NUKE_TEMP_DIR")
    if env_temp and env_temp.strip():
        p = Path(env_temp.strip()).resolve() / "AlphaGen"
        p.mkdir(parents=True, exist_ok=True)
        return p

    try:
        import tempfile
        p = Path(tempfile.gettempdir()).resolve() / "nuke" / "AlphaGen"
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        pass

    fallback = ROOT / "jobs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def inputs_changed(node):
    nuke = _nuke()
    if nuke.thisKnob() and nuke.thisKnob().name() != "inputChange":
        return
    for index, name in (
        (1, "has_selection_input"),
        (2, "has_core_input"),
        (3, "has_garbage_input")
    ):
        if name in node.knobs():
            node[name].setValue(node.input(index) is not None)

    # When Source (input 0) is connected or changed, automatically detect frame range and alpha folder!
    if node.input(0) is not None:
        sync_source_range(node)
        alpha_dir = get_source_alpha_dir(node)
        if alpha_dir:
            set_status(node, f"Source connected. Alpha will save to: {alpha_dir}")


def set_status(node, message):
    if "ag_status_label" in node.knobs():
        node["ag_status_label"].setValue(message)


def update_progress_ui(node, progress_val, message):
    """Update progress bar HTML and status label."""
    pct = max(0, min(100, int(progress_val * 100)))
    if pct > 0:
        bar_html = (
            f"<div style='background-color:#222; border-radius:4px; height:20px; border:1px solid #444; width:100%;'>"
            f"<div style='background:linear-gradient(90deg, #1976D2, #4CAF50); width:{pct}%; height:100%; "
            f"border-radius:3px; text-align:center; color:white; font-size:11px; font-weight:bold; line-height:20px;'>"
            f"{pct}% ({message})</div></div>"
        )
    else:
        bar_html = (
            f"<div style='background-color:#222; border-radius:4px; height:20px; border:1px solid #444; width:100%; "
            f"text-align:center; color:#888; font-size:11px; line-height:20px; font-weight:bold;'>"
            f"{message}</div>"
        )
    if "ag_progress_bar" in node.knobs():
        node["ag_progress_bar"].setValue(bar_html)
    set_status(node, message)


def start_job_monitor(node, job_path, job_mode):
    """Background monitor using QTimer that auto-updates progress and loads results."""
    if not QtCore:
        return
    node_name = node.fullName()
    if node_name in ACTIVE_TIMERS:
        try:
            ACTIVE_TIMERS[node_name].stop()
        except Exception:
            pass

    timer = QtCore.QTimer()
    ACTIVE_TIMERS[node_name] = timer
    status_file = Path(job_path) / "status.json"

    def check_status():
        try:
            nuke = _nuke()
            curr_node = nuke.toNode(node_name)
            if not curr_node:
                timer.stop()
                ACTIVE_TIMERS.pop(node_name, None)
                return

            if status_file.is_file():
                try:
                    data = json.loads(status_file.read_text(encoding="utf-8"))
                except Exception:
                    return
                state = data.get("state", "queued")
                prog = float(data.get("progress", 0.0))
                msg = data.get("message", state)
                update_progress_ui(curr_node, prog, msg)

                if state == "complete":
                    timer.stop()
                    ACTIVE_TIMERS.pop(node_name, None)
                    update_progress_ui(curr_node, 1.0, "Complete!")
                    res_path = data.get("result")
                    if res_path and Path(res_path).is_file():
                        res_data = json.loads(Path(res_path).read_text(encoding="utf-8"))
                        if job_mode == "vlm" or "sam2_iou" in res_data:
                            handle_vlm_complete(curr_node, res_data)
                        else:
                            load_result(curr_node, res_path)

                elif state in ("failed", "cancelled"):
                    timer.stop()
                    ACTIVE_TIMERS.pop(node_name, None)
                    err_msg = data.get("message", "Job failed")
                    update_progress_ui(curr_node, 0.0, f"Error: {err_msg}")
                    nuke.message(f"AlphaGen Job {state.capitalize()}:\n{err_msg}")
        except Exception:
            pass

    timer.timeout.connect(check_status)
    timer.start(500)


def handle_vlm_complete(node, res_data):
    mask_p = res_data["mask_path"]
    if "vlm_mask_path" in node.knobs():
        node["vlm_mask_path"].setValue(mask_p)
    exp_ratio = res_data.get("expand_ratio", 1.2)
    exp_info = f", Expand: {exp_ratio:.1f}x"
    if "ag_target_info" in node.knobs():
        node["ag_target_info"].setValue(
            f"Detected: {res_data.get('target_description', '')} (IoU: {res_data.get('sam2_iou', 1.0):.2f}{exp_info})"
        )
    load_vlm_preview(node, mask_p)
    node["view_mode"].setValue("Target Selection (VLM)")
    set_status(node, f"Target Ready: {res_data.get('target_description', '')} (Expanded {exp_ratio:.1f}x)")


def _is_pid_running(pid):
    if not pid:
        return False
    if os.name == "nt":
        try:
            res = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
            return str(pid) in res.stdout
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _new_job(node, mode):
    old = node["job_path"].value() if "job_path" in node.knobs() else ""
    if old and (Path(old) / "status.json").is_file():
        state = _read(Path(old) / "status.json").get("state", "complete")
        if state in ("queued", "preparing", "generating", "importing", "refining", "analyzing"):
            pid_file = Path(old) / "worker.pid"
            is_alive = False
            if pid_file.is_file():
                try:
                    pid = int(pid_file.read_text(encoding="utf-8").strip())
                    is_alive = _is_pid_running(pid)
                except Exception:
                    pass
            if is_alive:
                raise RuntimeError("This node already has a running job. Click 'Stop Queue (중단)' or wait for completion.")
            else:
                try:
                    (Path(old) / "status.json").write_text(
                        json.dumps({"state": "cancelled", "message": "Previous job was aborted"}),
                        encoding="utf-8"
                    )
                except Exception:
                    pass
    cache_dir = get_nuke_cache_dir()
    job = cache_dir / (mode + "_" + uuid.uuid4().hex[:12])
    job.mkdir(parents=True, exist_ok=False)
    if "job_path" in node.knobs():
        node["job_path"].setValue(job.as_posix())
    return job


def _launch(node, job, request, job_mode):
    python = resolve_python()
    if not python.is_file():
        raise ValueError(f"AI Python executable not found: {python}")
    request["settings"] = _settings(node)
    (job / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    (job / "status.json").write_text(json.dumps({"state": "queued", "progress": 0.02, "message": "Starting worker..."}), encoding="utf-8")
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["ALPHAGEN_CACHE_DIR"] = str(job.parent)
    (job / "temp").mkdir(exist_ok=True)
    env["TMP"] = env["TEMP"] = str(job / "temp")
    try:
        log = (job / "worker.log").open("w", encoding="utf-8")
        proc = subprocess.Popen([str(python), "-u", str(ROOT / "alphagen_worker.py"), str(job / "request.json")],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        (job / "worker.pid").write_text(str(proc.pid), encoding="utf-8")
        ACTIVE_PROCESSES[node.fullName()] = proc
    except Exception as exc:
        (job / "status.json").write_text(json.dumps({"state": "failed", "message": str(exc)}), encoding="utf-8")
        raise

    update_progress_ui(node, 0.05, "Task dispatched...")
    start_job_monitor(node, job.as_posix(), job_mode)


def _export(upstream, path, first, last, mask=False):
    parent_name = upstream.fullName().rpartition(".")[0]
    context = _nuke().toNode(parent_name) if parent_name else _nuke().root()
    with context:
        return _export_in_context(upstream, path, first, last, mask)


def _export_in_context(upstream, path, first, last, mask=False):
    nuke = _nuke()
    temporary = []
    try:
        if mask:
            expression = nuke.nodes.Expression(inputs=[upstream])
            temporary.append(expression)
            for index in range(3):
                expression["expr" + str(index)].setValue("a")
            expression["expr3"].setValue("1")
            upstream = expression
        writer = nuke.nodes.Write(inputs=[upstream])
        temporary.append(writer)
        writer["file"].setValue(path.as_posix())
        writer["file_type"].setValue("png")
        writer["channels"].setValue("rgb")
        if "datatype" in writer.knobs():
            writer["datatype"].setValue("16 bit")
        if mask:
            writer["raw"].setValue(True)
        else:
            writer["colorspace"].setValue("sRGB")
        nuke.execute(writer, first, last)
    finally:
        for temporary_node in reversed(temporary):
            nuke.delete(temporary_node)


def analyze_vlm(node):
    """Analyze reference frame with VLM and segment target with SAM2."""
    if not node.input(0):
        raise ValueError("Connect Source RGB input to analyze target.")
    prompt = node["ag_prompt"].value().strip() if "ag_prompt" in node.knobs() else ""
    if not prompt:
        raise ValueError("Enter a Target Prompt describing the subject to segment.")

    ref_frame = int(node["ag_ref_frame"].value()) if "ag_ref_frame" in node.knobs() else int(node["first"].value())
    mask_expand = float(node["ag_mask_expand"].value()) if "ag_mask_expand" in node.knobs() else 1.2
    alpha_dir = get_source_alpha_dir(node)
    job = _new_job(node, "vlm")
    (job / "exports").mkdir(exist_ok=True)
    ref_img = job / "exports/ref_frame.png"

    update_progress_ui(node, 0.05, "Exporting reference frame...")
    _export(node.input(0), ref_img, ref_frame, ref_frame)

    request = {
        "mode": "vlm",
        "reference_frame": ref_img.as_posix(),
        "prompt": prompt,
        "mask_expand": mask_expand,
        "output_dir": alpha_dir,
    }
    _launch(node, job, request, job_mode="vlm")


def load_vlm_preview(node, mask_path):
    """Load the VLM SAM2 mask into the internal Read_selection node."""
    reader = node.node("Read_selection")
    if reader and Path(mask_path).is_file():
        clean_path = str(Path(mask_path).resolve()).replace("\\", "/")
        reader["file"].setValue(clean_path)
        reader["raw"].setValue(True)
        reader["on_error"].setValue("black")
        reader["reload"].execute()
    if "has_vlm_mask" in node.knobs():
        node["has_vlm_mask"].setValue(True)


def generate(node):
    nuke = _nuke()
    first, last = int(node["first"].value()), int(node["last"].value())
    count = last - first + 1
    if count < 9 or (count - 1) % 8:
        raise ValueError("Use 8n+1 frames (9, 17, 25, 33, 41, 49, 97, 121, 241...). LTX requires temporal factor 8n+1.")
    if not node.input(0):
        raise ValueError("Connect Source RGB input.")

    has_manual_selection = node.input(1) is not None
    prompt = node["ag_prompt"].value().strip() if "ag_prompt" in node.knobs() else ""
    vlm_mask = node["vlm_mask_path"].value().strip() if "vlm_mask_path" in node.knobs() else ""
    mask_expand = float(node["ag_mask_expand"].value()) if "ag_mask_expand" in node.knobs() else 1.2
    alpha_dir = get_source_alpha_dir(node)

    if not has_manual_selection and not prompt and not (vlm_mask and Path(vlm_mask).is_file()):
        raise ValueError("Provide either a Selection input or a Target Prompt for VLM.")

    job = _new_job(node, "generate")
    (job / "exports").mkdir()
    source = job / "exports/source.%06d.png"

    update_progress_ui(node, 0.05, "Exporting source sequence...")
    _export(node.input(0), source, first, last)

    if has_manual_selection:
        if format_signature(node.input(0)) != format_signature(node.input(1)):
            raise ValueError("Reformat Selection to match Source before generation")
        mask = job / "exports/selection.%06d.png"
        update_progress_ui(node, 0.1, "Exporting manual selection mask...")
        _export(node.input(1), mask, first, last, mask=True)
        mask_pattern = mask.as_posix()
    else:
        # Use VLM mask
        if not vlm_mask or not Path(vlm_mask).is_file():
            ref_frame = int(node["ag_ref_frame"].value()) if "ag_ref_frame" in node.knobs() else first
            ref_img = job / "exports/ref_frame.png"
            update_progress_ui(node, 0.08, "Exporting reference frame for VLM grounding...")
            _export(node.input(0), ref_img, ref_frame, ref_frame)
            mask_pattern = ""
        else:
            mask_pattern = vlm_mask

    request = {
        "mode": "generate",
        "source_pattern": source.as_posix(),
        "mask_pattern": mask_pattern,
        "first": first,
        "last": last,
        "fps": float(nuke.root()["fps"].value()),
        "max_side": int(node["max_side"].value()),
        "seed": int(node["seed"].value()),
        "prompt": prompt,
        "mask_expand": mask_expand,
        "output_dir": alpha_dir,
    }
    if not mask_pattern and not has_manual_selection:
        request["reference_frame"] = ref_img.as_posix()
    _launch(node, job, request, job_mode="generate")


def refine_cached(node):
    result_path = node["result_path"].value() if "result_path" in node.knobs() else ""
    if not result_path or not Path(result_path).is_file():
        raise ValueError("Import or generate and load an alpha first before refining")
    job = _new_job(node, "refine")
    alpha_dir = get_source_alpha_dir(node)
    update_progress_ui(node, 0.05, "Re-baking matte refinement on CPU...")
    _launch(node, job, {"mode": "refine", "result": result_path, "output_dir": alpha_dir}, job_mode="refine")


def load_result(node, path):
    configure_graph(node)
    data = _read(path)
    for name in ("raw", "final"):
        pattern = data.get(name + "_pattern", "")
        for frame in range(data["first"], data["last"] + 1):
            if not Path(pattern % frame).is_file():
                raise ValueError(f"Missing {name} frame {frame}")
    for name in ("raw", "final", "core", "edge"):
        reader = node.node("Read_" + name)
        pattern = data.get(name + "_pattern", "")
        if reader and pattern:
            first_frame_file = Path(pattern % data["first"]) if "%" in pattern else Path(pattern)
            if first_frame_file.is_file():
                reader["file"].setValue(pattern)
                reader["first"].setValue(data["first"])
                reader["last"].setValue(data["last"])
                reader["origfirst"].setValue(data["first"])
                reader["origlast"].setValue(data["last"])
                reader["raw"].setValue(True)
                reader["on_error"].setValue("black")
                reader["reload"].execute()
            else:
                reader["file"].setValue("")
                reader["on_error"].setValue("black")
    if "result_path" in node.knobs():
        node["result_path"].setValue(str(Path(path).resolve()).replace("\\", "/"))
    if data.get("vlm_mask_path") and Path(data["vlm_mask_path"]).is_file():
        if "vlm_mask_path" in node.knobs():
            node["vlm_mask_path"].setValue(data["vlm_mask_path"])
        load_vlm_preview(node, data["vlm_mask_path"])
    node["has_result"].setValue(True)
    node["first"].setValue(data["first"])
    node["last"].setValue(data["last"])
    message = f"Ready — {data['first']}–{data['last']}, {data['size'][0]}x{data['size'][1]}"
    set_status(node, message)
    update_progress_ui(node, 1.0, message)
    for key, value in data.get("settings", {}).items():
        if key in node.knobs():
            node[key].setValue(value)


def cancel(node):
    node_name = node.fullName()
    job_str = node["job_path"].value() if "job_path" in node.knobs() else ""
    job = Path(job_str).resolve() if job_str else None

    # 1. Stop background UI QTimer monitor
    if node_name in ACTIVE_TIMERS:
        try:
            ACTIVE_TIMERS[node_name].stop()
        except Exception:
            pass
        ACTIVE_TIMERS.pop(node_name, None)

    # 2. Write cancel request file for graceful checkpoint exit
    if job and job.is_dir():
        try:
            (job / "cancel.request").write_text("cancel", encoding="utf-8")
        except Exception:
            pass

    # 3. Terminate running worker process tree immediately
    proc = ACTIVE_PROCESSES.pop(node_name, None)
    pid = None
    if proc:
        pid = proc.pid
    elif job and (job / "worker.pid").is_file():
        try:
            pid = int((job / "worker.pid").read_text(encoding="utf-8").strip())
        except Exception:
            pass

    killed = False
    if pid and _is_pid_running(pid):
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                import signal
                os.kill(pid, signal.SIGKILL)
            killed = True
        except Exception:
            pass

    # 4. Mark status.json as cancelled to unblock next jobs immediately
    if job and job.is_dir():
        try:
            (job / "status.json").write_text(
                json.dumps({"state": "cancelled", "progress": 0.0, "message": "Queue stopped by user"}, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass

    # 5. Update UI progress and status
    msg = "⏹ Queue Terminated by User" if killed else "⏹ Queue Stopped"
    update_progress_ui(node, 0.0, msg)
    set_status(node, "Queue cancelled by user. Ready for next task.")


def action(node, name):
    try:
        {
            "vlm": analyze_vlm,
            "generate": generate,
            "refine": refine_cached,
            "cancel": cancel,
            "stop": cancel,
        }[name](node)
    except Exception as exc:
        set_status(node, "Error: " + str(exc))
        update_progress_ui(node, 0.0, f"Error: {exc}")
        _nuke().message("AlphaGen\n" + str(exc))


def register():
    nuke = _nuke()
    nuke.pluginAddPath(str(ROOT / "gizmos"))
    toolbar = nuke.menu("Nodes")
    menu = toolbar.findItem("AlphaGen") or toolbar.addMenu("AlphaGen")
    if not menu.findItem("AlphaGen"):
        menu.addCommand("AlphaGen", "import alphagen_nuke; alphagen_nuke.create()")


def demo():
    """Create test graph without replacing an open script."""
    nuke = _nuke()
    result = ROOT / "experiments/demo_refine/result.json"
    if not result.is_file():
        result = ROOT / "jobs/demo_refine/result.json"
    if not result.is_file():
        raise ValueError("Demo fixture not found in experiments/demo_refine")
    source = nuke.nodes.Read(file=(ROOT / "experiments/001/input.avi").as_posix())
    source["first"].setValue(1)
    source["last"].setValue(17)
    source["colorspace"].setValue("sRGB")
    selection = nuke.nodes.Read(file=(ROOT / "experiments/002/initial_mask.png").as_posix())
    selection["raw"].setValue(True)
    expression = nuke.nodes.Expression(inputs=[selection], expr3="r")
    node = create()
    node.setInput(0, source)
    node.setInput(1, expression)
    load_result(node, result)
    node["view_mode"].setValue("Final alpha")
    return node
