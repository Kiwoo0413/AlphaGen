"""External worker: Nuke export -> LTX, or cached alpha -> CPU refinement.

All job writes are restricted to AlphaGen/jobs. No model download or installation.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
from alphagen_matte import Settings, refine


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    temporary = path.with_suffix(".tmp")
    for attempt in range(10):
        try:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)
            return
        except (PermissionError, OSError) as exc:
            winerror = getattr(exc, "winerror", None)
            if winerror in (5, 32) or isinstance(exc, PermissionError):
                time.sleep(0.05)
            else:
                raise
    try:
        path.write_text(payload, encoding="utf-8")
    except Exception:
        if path.name == "status.json":
            return
        raise


def job_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to((ROOT / "jobs").resolve()):
        raise ValueError("Jobs must be inside AlphaGen/jobs")
    return path


def resolve_python(path_str=None):
    if path_str and path_str != "auto":
        p = Path(path_str)
        if p.is_file():
            return str(p)
    env_py = os.environ.get("ALPHAGEN_PYTHON")
    if env_py and Path(env_py).is_file():
        return str(Path(env_py))
    for c in [
        Path("D:/AI/ComfyUI_DATA/.venv/Scripts/python.exe"),
        ROOT / ".venv/Scripts/python.exe",
        ROOT / ".venv/bin/python",
    ]:
        if c.is_file():
            return str(c)
    which_py = shutil.which("python")
    if which_py and "nuke" not in which_py.lower():
        return which_py
    return sys.executable


def status(job, state, message, progress=None, **extra):
    data = {"state": state, "message": message, **extra}
    if progress is not None:
        data["progress"] = float(progress)
    write_json(job / "status.json", data)


def cancelled(job):
    if (job / "cancel.request").exists():
        raise InterruptedError("Cancelled by user")


def save_alpha(path, alpha):
    data = np.round(np.clip(alpha, 0, 1) * 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), data):
        raise RuntimeError("Cannot write " + str(path))


def read_alpha(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Cannot read " + str(path))
    if image.ndim == 3:
        image = image[:, :, 0]
    scale = 65535 if image.dtype == np.uint16 else 255 if image.dtype == np.uint8 else 1
    return image.astype(np.float32) / scale


def cached_alpha(pt_path):
    import torch
    tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise ValueError("Decoded cache must have shape T,H,W,3")
    # Official decoder cache is floating RGB in [0,1], before video compression.
    array = tensor.float().numpy()
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("Decoded cache must be finite RGB in [0,1]")
    return array.mean(axis=-1)


def export_raw(job, alpha, first, content_size=None, output_size=None, output_dir=None):
    target = Path(output_dir) if output_dir else job
    directory = target / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    job_directory = job / "raw"
    job_directory.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(alpha):
        cancelled(job)
        if content_size:
            w, h = content_size
            image = image[:h, :w]
        if output_size and (image.shape[1], image.shape[0]) != tuple(output_size):
            image = cv2.resize(image, tuple(output_size), interpolation=cv2.INTER_LINEAR)
        save_alpha(directory / f"alpha.{first + index:06d}.png", image)
        if target != job:
            save_alpha(job_directory / f"alpha.{first + index:06d}.png", image)
    return {"raw_pattern": (directory / "alpha.%06d.png").as_posix(),
            "first": first, "last": first + len(alpha) - 1,
            "size": [image.shape[1], image.shape[0]], "precision": "16-bit PNG, linear alpha"}


def bake(job, metadata, settings, output_dir=None):
    s = Settings(**settings)
    s.validate()
    target = Path(output_dir) if output_dir else job
    target.mkdir(parents=True, exist_ok=True)
    for name in ("final", "core", "edge"):
        (target / name).mkdir(parents=True, exist_ok=True)
        if target != job:
            (job / name).mkdir(parents=True, exist_ok=True)
    metrics = []
    start = time.monotonic()
    total_frames = max(1, metadata["last"] - metadata["first"] + 1)
    for frame in range(metadata["first"], metadata["last"] + 1):
        cancelled(job)
        if frame == metadata["first"] or frame == metadata["last"] or (frame - metadata["first"]) % 5 == 0:
            prog = 0.85 + 0.14 * (frame - metadata["first"]) / total_frames
            status(job, "refining", f"Refining frame {frame}/{metadata['last']}", progress=prog)
        raw = read_alpha(metadata["raw_pattern"] % frame)
        output = refine(raw, s)
        # 1. Primary final alpha directly in target / alpha.XXXXXX.png
        save_alpha(target / f"alpha.{frame:06d}.png", output["final"])
        # 2. Layer subdirectories (final, core, edge)
        for name in ("final", "core", "edge"):
            save_alpha(target / name / f"alpha.{frame:06d}.png", output[name])
            if target != job:
                save_alpha(job / name / f"alpha.{frame:06d}.png", output[name])
        metrics.append({"frame": frame,
                        "raw_mean": float(raw.mean()), "final_mean": float(output["final"].mean()),
                        "core_pixels": int((output["core"] >= 0.999).sum()),
                        "raw_background_below_clip_mean": float(raw[raw <= s.black_clip].mean()) if (raw <= s.black_clip).any() else None,
                        "final_background_below_clip_mean": float(output["final"][raw <= s.black_clip].mean()) if (raw <= s.black_clip).any() else None})
    result = dict(metadata)
    result["final_pattern"] = (target / "alpha.%06d.png").as_posix()
    result["core_pattern"] = (target / "core" / "alpha.%06d.png").as_posix()
    result["edge_pattern"] = (target / "edge" / "alpha.%06d.png").as_posix()
    result.update(settings=asdict(s), refinement_seconds=time.monotonic() - start,
                  metrics=metrics, job=job.as_posix(), output_dir=target.as_posix())
    write_json(job / "result.json", result)
    if target != job:
        write_json(target / "result.json", result)
    status(job, "complete", "Ready to load", progress=1.0, result=(target / "result.json").as_posix())


def run_inference(job, config):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(ROOT / p) for p in (
        "runtime", "vendor/LTX-2/packages/ltx-core/src", "vendor/LTX-2/packages/ltx-pipelines/src"))
    py_exe = resolve_python(config.get("python"))
    command = [py_exe, str(ROOT / "tools/run_experiment.py"),
               "--config", str(job / "config.json"), "--case", "generic_alpha", "--run"]
    status(job, "generating", "Starting LTX Video Diffusion...", progress=0.2)
    with (job / "inference.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            import re
            last_check = 0
            while process.poll() is None:
                cancelled(job)
                time.sleep(1.0)
                now = time.monotonic()
                if now - last_check > 2.0:
                    last_check = now
                    try:
                        log_file = job / "inference.log"
                        if log_file.is_file():
                            text = log_file.read_text(encoding="utf-8", errors="ignore")
                            matches = re.findall(r"(\d+)/(\d+)\s+\[", text)
                            if matches:
                                cur_step, max_step = map(int, matches[-1])
                                if max_step > 0:
                                    step_prog = 0.2 + 0.65 * (cur_step / max_step)
                                    status(job, "generating", f"LTX Denoising step {cur_step}/{max_step}", progress=step_prog)
                    except Exception:
                        pass
        except BaseException:
            # The runner owns a pipeline child. On Windows terminate this job's tree only.
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                process.terminate()
            process.wait()
            raise
    if process.returncode:
        raise RuntimeError("LTX failed; see results/generic_alpha.log and inference.log")


@contextmanager
def gpu_lock():
    """OS lock is automatically released if the worker exits or crashes."""
    with (ROOT / "jobs/gpu.lock").open("a+b") as lock:
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another AlphaGen generation is using the GPU") from exc
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def prepare_input(job, request):
    first, last = request["first"], request["last"]
    count = last - first + 1
    if count < 9 or count > 121 or (count - 1) % 8:
        raise ValueError("Frame count must be 8n+1, from 9 to 121")
    fps = float(request["fps"])
    side = int(request.get("max_side", 512))
    if not math.isfinite(fps) or fps <= 0 or side < 64 or side > 1024:
        raise ValueError("Invalid fps or max side (64..1024)")
    source_pattern = request["source_pattern"]
    first_image = cv2.imread(source_pattern % first, cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError("Cannot decode exported source")
    h, w = first_image.shape[:2]
    scale = min(1, side / max(w, h))
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    pw, ph = math.ceil(rw / 32) * 32, math.ceil(rh / 32) * 32
    paths = [job / "input.avi", job / "attention_mask.avi"]
    writers = [cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"FFV1"), fps, (pw, ph)) for p in paths]
    if not all(writer.isOpened() for writer in writers):
        for writer in writers:
            writer.release()
        raise RuntimeError("Lossless FFV1 writer unavailable")
    try:
        for frame in range(first, last + 1):
            cancelled(job)
            image = cv2.imread(source_pattern % frame, cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (h, w):
                raise ValueError(f"Source dimensions mismatch at {frame}")
            pattern = request["mask_pattern"]
            mask = read_alpha(pattern % frame if "%" in pattern else pattern)
            if mask.shape != (h, w):
                raise ValueError("Selection mask must match source format")
            if frame == first and not (mask.max() > 0.01 and mask.min() < 0.99):
                raise ValueError("Selection mask is empty or covers the entire frame")
            canvas = np.zeros((ph, pw, 3), np.uint8)
            canvas[:rh, :rw] = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA)
            writers[0].write(canvas)
            mask_canvas = np.zeros((ph, pw), np.uint8)
            mask_canvas[:rh, :rw] = np.round(cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_LINEAR) * 255).astype(np.uint8)
            writers[1].write(cv2.cvtColor(mask_canvas, cv2.COLOR_GRAY2BGR))
    finally:
        for writer in writers:
            writer.release()
    for path in paths:
        reader = cv2.VideoCapture(str(path))
        decoded = 0
        while reader.read()[0]:
            decoded += 1
        reader.release()
        if decoded != count:
            raise RuntimeError("Prepared video frame count mismatch")
    manifest = {"prepared_video": paths[0].as_posix(), "prepared_size": [pw, ph],
                "resized_content_size": [rw, rh], "source_size": [w, h],
                "source_fps": fps, "frames": count, "seed": int(request.get("seed", 1234))}
    write_json(job / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    job = job_path(args.request.parent)
    try:
        (job / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass
    request = read_json(args.request)
    started = time.monotonic()
    try:
        cancelled(job)
        mode = request["mode"]
        output_dir = request.get("output_dir")
        if mode == "vlm":
            status(job, "analyzing", "Running VLM grounding & SAM2 segmentation...")
            from alphagen_vlm import analyze_and_segment
            ref_frame = request["reference_frame"]
            prompt = request["prompt"]
            expand_ratio = float(request.get("mask_expand", 1.2))
            target_dir = Path(output_dir) if output_dir else job
            target_dir.mkdir(parents=True, exist_ok=True)
            out_mask = target_dir / "vlm_initial_mask.png"
            vlm_res = analyze_and_segment(ref_frame, prompt, output_mask_path=out_mask, expand_ratio=expand_ratio)
            vlm_res["mask_pattern"] = out_mask.as_posix()
            vlm_res["job"] = job.as_posix()
            vlm_res["output_dir"] = target_dir.as_posix()
            write_json(job / "result.json", vlm_res)
            if target_dir != job:
                write_json(target_dir / "vlm_selection.json", vlm_res)
            target_lbl = vlm_res.get("target_description") or prompt
            iou_val = vlm_res.get("sam2_iou", 1.0)
            status(job, "complete", f"Target identified: {target_lbl} (IoU: {iou_val:.2f}, Expand: {expand_ratio:.1f}x)", result=(job / "result.json").as_posix())
            return
        elif mode == "refine":
            metadata = read_json(request["result"])
            output_dir = output_dir or metadata.get("output_dir")
        elif mode == "import":
            status(job, "importing", "Loading decoded LTX tensor")
            manifest = read_json(request["manifest"])
            alpha = cached_alpha(request["decoded_cache"])
            if len(alpha) != manifest["frames"]:
                raise ValueError("Cache and manifest frame counts differ")
            # Imported experiment uses prepared resolution, with right/bottom padding removed.
            metadata = export_raw(job, alpha, request.get("first", 1), manifest["resized_content_size"], output_dir=output_dir)
            metadata.update(source_manifest=request["manifest"], source_cache=request["decoded_cache"],
                            resolution_note="Prepared experiment resolution; not native source detail")
        elif mode == "generate":
            mask_pattern = request.get("mask_pattern", "")
            target_dir = Path(output_dir) if output_dir else job
            target_dir.mkdir(parents=True, exist_ok=True)
            vlm_mask_path = None
            if not mask_pattern or not Path(mask_pattern).is_file():
                status(job, "analyzing", "Analyzing target with VLM & SAM2...")
                from alphagen_vlm import analyze_and_segment
                ref_frame = request.get("reference_frame")
                if not ref_frame or not Path(ref_frame).is_file():
                    ref_frame = request["source_pattern"] % request["first"]
                prompt = request.get("prompt", "")
                expand_ratio = float(request.get("mask_expand", 1.2))
                out_mask = target_dir / "vlm_initial_mask.png"
                vlm_res = analyze_and_segment(ref_frame, prompt, output_mask_path=out_mask, expand_ratio=expand_ratio)
                mask_pattern = out_mask.as_posix()
                request["mask_pattern"] = mask_pattern
                vlm_mask_path = mask_pattern
                if "target_description" not in request:
                    request["target_description"] = vlm_res.get("target_description")
            status(job, "preparing", "Preparing RGB and selection mask")
            manifest = prepare_input(job, request)
            config = read_json(ROOT / "experiment_config.json")
            config.update(manifest=(job / "manifest.json").as_posix(),
                          conditioning_attention_mask={"path": (job / "attention_mask.avi").as_posix(), "strength": 1.0})
            if request.get("prompt") or request.get("target_description"):
                desc = request.get("target_description") or request.get("prompt")
                config["prompt"] = (
                    f"Generate a grayscale alpha matte only for {desc}. "
                    "Maintain the same subject's identity throughout the clip. "
                    "Preserve fine hair detail and soft edges. Exclude other subjects and the background. "
                    "White represents the selected subject, black represents excluded regions, and gray represents partial opacity."
                )
            write_json(job / "config.json", config)
            with gpu_lock():
                run_inference(job, config)
            alpha = cached_alpha(job / "results/generic_alpha.decoded.0.pt")
            if len(alpha) != manifest["frames"]:
                raise ValueError("Generated alpha frame count mismatch")
            metadata = export_raw(job, alpha, request["first"], manifest["resized_content_size"], manifest["source_size"], output_dir=output_dir)
            metadata.update(source_manifest=(job / "manifest.json").as_posix(),
                            resolution_note="LTX working resolution upscaled to source; no new detail recovered")
        else:
            raise ValueError("Unknown job mode")
        bake(job, metadata, request.get("settings", {}), output_dir=output_dir)
        res_file = (Path(output_dir) / "result.json") if output_dir else (job / "result.json")
        result = read_json(res_file)
        if vlm_mask_path:
            result["vlm_mask_path"] = vlm_mask_path
        result["job_seconds"] = time.monotonic() - started
        write_json(res_file, result)
        if output_dir and Path(output_dir) != job:
            write_json(job / "result.json", result)
    except BaseException as exc:
        status(job, "cancelled" if isinstance(exc, InterruptedError) else "failed", str(exc))
        raise


if __name__ == "__main__":
    main()
