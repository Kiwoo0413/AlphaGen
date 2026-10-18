"""Plan or run controlled Alpha Gen prompt experiments using an existing LTX runtime."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_path(path):
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    models_dir = os.environ.get("ALPHAGEN_MODELS_DIR")
    if models_dir:
        cand = Path(models_dir) / p
        if cand.exists():
            return cand
    for drive in ["E:/models", "D:/models", "C:/models", "./models"]:
        cand = Path(drive) / p
        if cand.exists():
            return cand
    return ROOT / p


def resolve_python(path_str):
    if path_str and path_str != "auto":
        p = Path(path_str)
        if p.is_file():
            return p
    env_py = os.environ.get("ALPHAGEN_PYTHON")
    if env_py and Path(env_py).is_file():
        return Path(env_py)
    candidates = [
        Path("D:/AI/ComfyUI_DATA/.venv/Scripts/python.exe"),
        Path(sys.executable),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return Path(sys.executable)


def absolute(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiment_config.json")
    parser.add_argument("--case", choices=["baseline", "selected", "target_a", "target_b", "generic_alpha"], default="baseline")
    parser.add_argument("--run", action="store_true", help="Actually launch inference; default only saves a plan")
    parser.add_argument("--retry-failed", action="store_true", help="Archive a failed attempt before retrying")
    args = parser.parse_args()
    config = read(args.config)
    manifest = read(absolute(config["manifest"]))
    selection = read(absolute(config["selection"]))
    case = selection["selected_id"] if args.case == "selected" else args.case
    candidates = {x["id"]: x for x in selection["candidates"]}
    if config.get("prompt"):
        prompt = config["prompt"]
    elif case == "generic_alpha":
        prompt = read(ROOT / "experiments/001/followup_conditions.json")["generic_alpha"]["prompt"]
    else:
        prompt = "" if case == "baseline" else candidates[case]["prompt"]
    if case != "baseline" and not prompt.strip():
        parser.error("Target prompt must not be empty")
    width, height = manifest["prepared_size"]
    if width % 32 or height % 32 or (manifest["frames"] - 1) % 8:
        parser.error("Prepared input does not satisfy model dimensions/frame grid")
    output_dir = absolute(config["manifest"]).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / f"{case}.mp4"
    missing = []
    py_exe = resolve_python(config.get("python"))
    command = [str(py_exe)]
    if config.get("pipeline_script"):
        script = absolute(config["pipeline_script"])
        if not script.is_file():
            missing.append("pipeline script")
        command.append(str(script))
    else:
        command.extend(["-m", "ltx_pipelines.ic_lora"])
    if not py_exe.is_file():
        missing.append("python runtime")
    for flag, path in config["models"].items():
        resolved = str(resolve_model_path(path))
        if not Path(resolved).is_file():
            missing.append({"component": flag, "path": resolved})
        command.extend([f"--{flag}", resolved])
        if flag == "lora":
            command.append("1.0")
    video = absolute(manifest["prepared_video"])
    if not video.is_file():
        missing.append("prepared video")
    command.extend([
        "--prompt", prompt,
        "--width", str(width * 2), "--height", str(height * 2),
        "--num-frames", str(manifest["frames"]),
        "--frame-rate", str(manifest["source_fps"]),
        "--video-conditioning", str(video), "1.0",
        "--skip-stage-2", "--seed", str(manifest["seed"]),
        "--offload", config["offload"], "--quantization", config["quantization"],
        "--output-path", str(output),
    ])
    chunk_frames = config.get("chunk_pixel_frames", 49)
    if chunk_frames and manifest["frames"] > chunk_frames:
        carry = config.get("chunk_carry_frames", 17)
        blend = config.get("chunk_blend_frames", carry)
        command.extend([
            "--chunked",
            "--chunk-pixel-frames", str(chunk_frames),
            "--chunk-carry-frames", str(carry),
            "--chunk-blend-frames", str(blend),
        ])
    mask_condition = config.get("conditioning_attention_mask")
    if mask_condition:
        mask_path = absolute(mask_condition["path"])
        strength = float(mask_condition["strength"])
        if not mask_path.is_file():
            missing.append("conditioning attention mask video")
        if not 0 < strength <= 1:
            parser.error("Mask conditioning strength must be in (0,1]")
        command.extend(["--conditioning-attention-mask", str(mask_path), str(strength)])
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "case": case,
        "prompt": prompt, "official_prompt_mode": case == "baseline",
        "status": "blocked_missing_components" if missing else "planned",
        "missing": missing, "command_argv": command,
        "conditioning_attention_mask": mask_condition,
        "note": "File readiness is not a guarantee of VRAM fit or runtime compatibility.",
    }
    plan = output_dir / f"{case}.plan.json"
    plan.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if missing:
        raise SystemExit(2)
    if not args.run:
        return
    run_report = output_dir / f"{case}.run.json"
    if run_report.exists() and args.retry_failed and not output.exists():
        if read(run_report).get("status") != "failed":
            parser.error("Only failed attempts may be retried")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        run_report.rename(output_dir / f"{case}.{stamp}.run.json")
        old_log = output_dir / f"{case}.log"
        if old_log.exists():
            old_log.rename(output_dir / f"{case}.{stamp}.log")
    if output.exists() or run_report.exists():
        parser.error("Result already exists; use a new experiment directory")
    report["status"] = "running"
    run_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    start = time.monotonic()
    with (output_dir / f"{case}.log").open("w", encoding="utf-8") as log:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(str(ROOT / p) for p in [
            "runtime", "vendor/LTX-2/packages/ltx-core/src", "vendor/LTX-2/packages/ltx-pipelines/src"
        ])
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    report.update(exit_code=result.returncode, elapsed_seconds=time.monotonic() - start)
    report["status"] = "output_created_unreviewed" if result.returncode == 0 and output.is_file() else "failed"
    run_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(result.returncode if result.returncode else (0 if output.is_file() else 1))


if __name__ == "__main__":
    main()
