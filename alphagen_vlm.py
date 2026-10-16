"""AlphaGen Vision-Language Model (VLM) & SAM2 Grounding Module.

Given an input image and a natural language description (e.g. '중앙의 갈색 셔츠를 입은 인물'):
1. Queries the Vision-Language Model (Gemini Flash) to extract the spatial bounding box [ymin, xmin, ymax, xmax]
   and semantic target attributes.
2. Feeds the grounded bounding box into the local SAM2.1 model (Hiera-Large) to produce
   a pixel-accurate binary segmentation matte on the reference frame.
3. Packages the target condition tuple (description, reference frame, mask path, bbox) for LTX Alpha conditioning.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_SAM2_CHECKPOINT = Path(
    "C:/Users/yido2/.cache/huggingface/hub/models--facebook--sam2.1-hiera-large/snapshots/665f8e2ad61cf5f53d65644ff27c8ee525124610"
)

# Supported Gemini models in order of priority
GEMINI_MODELS = [
    "models/gemini-3.5-flash-lite",
    "models/gemini-flash-latest",
    "models/gemini-3.8-flash",
]


def ground_object_bbox_vlm(
    image_bgr: np.ndarray,
    target_prompt: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Query Gemini VLM to ground the target object into a 2D bounding box (0-1000 scale)."""
    import requests

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    h, w = image_bgr.shape[:2]
    scale = min(1.0, 768 / max(w, h))
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image_bgr, (rw, rh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_img = base64.b64encode(buf).decode("utf-8")

    sys_prompt = f"""You are an expert Vision-Language Grounding Model for VFX compositing and matting.
Locate the specific target object described by the user in this video frame.
Target Object: "{target_prompt}"

Return ONLY a raw JSON object (no markdown, no backticks, no extra text):
{{
  "target_description": "concise description of the identified target object in English",
  "box_2d": [ymin, xmin, ymax, xmax],  // normalized integer coordinates in [0, 1000] scale
  "center_point": [x, y], // normalized integer coordinates in [0, 1000] scale
  "confidence": 0.95
}}
"""
    payload = {
        "contents": [{
            "parts": [
                {"text": sys_prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1
        }
    }

    last_error = ""
    for model_name in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
        try:
            resp = requests.post(url, json=payload, timeout=25)
            if resp.status_code == 200:
                res_json = resp.json()
                text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Clean any accidental markdown wrap
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                data = json.loads(text)
                data["vlm_model"] = model_name
                return data
            else:
                last_error = f"{model_name}: HTTP {resp.status_code} - {resp.text[:150]}"
        except Exception as exc:
            last_error = f"{model_name}: {exc}"

    raise RuntimeError(f"VLM grounding failed across models. Last error: {last_error}")


def segment_with_sam2(
    image_bgr: np.ndarray,
    box_2d_1000: list[int],
    checkpoint_path: Path = DEFAULT_SAM2_CHECKPOINT,
) -> Tuple[np.ndarray, float]:
    """Segment target object using SAM2 given a 0-1000 normalized bounding box."""
    import torch
    from transformers import Sam2Model, Sam2Processor

    h, w = image_bgr.shape[:2]
    ymin, xmin, ymax, xmax = box_2d_1000
    box_xyxy = [
        int(xmin / 1000 * w),
        int(ymin / 1000 * h),
        int(xmax / 1000 * w),
        int(ymax / 1000 * h),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam2Processor.from_pretrained(str(checkpoint_path), local_files_only=True)
    model = Sam2Model.from_pretrained(str(checkpoint_path), local_files_only=True).to(device).eval()

    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_img, input_boxes=[[box_xyxy]], return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)

    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
    mask_binary = (masks[0, 0].numpy() > 0).astype(np.uint8) * 255
    iou_score = float(outputs.iou_scores[0, 0].item()) if hasattr(outputs, "iou_scores") else 1.0

    return mask_binary, iou_score


def expand_mask(mask: np.ndarray, expand_ratio: float = 1.2) -> np.ndarray:
    """Expand binary or grayscale alpha mask by expand_ratio (e.g. 1.2x = 20% margin) around object perimeter.

    Uses an elliptical morphological structuring element with radii proportional to the
    bounding dimensions of the object, scaling width and height organically by expand_ratio.
    """
    if expand_ratio <= 1.0:
        return mask.copy()

    is_float = np.issubdtype(mask.dtype, np.floating)
    binary = (mask > (0.01 if is_float else 2)).astype(np.uint8)
    pts = cv2.findNonZero(binary)
    if pts is None:
        return mask.copy()

    _, _, w, h = cv2.boundingRect(pts)
    margin_ratio = (expand_ratio - 1.0) / 2.0
    rx = max(1, round(w * margin_ratio))
    ry = max(1, round(h * margin_ratio))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))
    expanded = cv2.dilate(mask, kernel)
    return expanded


def analyze_and_segment(
    image_path: str | Path,
    target_prompt: str,
    output_mask_path: Optional[str | Path] = None,
    checkpoint_path: Path = DEFAULT_SAM2_CHECKPOINT,
    expand_ratio: float = 1.2,
) -> Dict[str, Any]:
    """End-to-end: input image + text prompt -> VLM detection -> SAM2 mask generation -> ~1.2x auto expansion."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image at {image_path}")

    start_time = time.monotonic()
    # 1. Grounding with VLM
    vlm_result = ground_object_bbox_vlm(img, target_prompt)
    box_2d = vlm_result["box_2d"]

    # 2. Segmenting with SAM2
    raw_mask, iou = segment_with_sam2(img, box_2d, checkpoint_path=checkpoint_path)

    # 3. Automatically expand surrounding alpha by expand_ratio (default 1.2x)
    mask = expand_mask(raw_mask, expand_ratio=expand_ratio)

    h, w = img.shape[:2]
    ymin, xmin, ymax, xmax = box_2d
    box_xyxy = [
        int(xmin / 1000 * w),
        int(ymin / 1000 * h),
        int(xmax / 1000 * w),
        int(ymax / 1000 * h),
    ]

    out_mask_path = Path(output_mask_path) if output_mask_path else image_path.parent / "vlm_initial_mask.png"
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_mask_path), mask)

    elapsed = time.monotonic() - start_time
    summary = {
        "prompt": target_prompt,
        "target_description": vlm_result.get("target_description", target_prompt),
        "vlm_model": vlm_result.get("vlm_model", "gemini"),
        "confidence": vlm_result.get("confidence", 1.0),
        "box_2d_norm": box_2d,
        "box_xyxy_pixel": box_xyxy,
        "sam2_iou": iou,
        "expand_ratio": expand_ratio,
        "raw_foreground_ratio": float((raw_mask > 0).mean()),
        "foreground_ratio": float((mask > 0).mean()),
        "mask_path": out_mask_path.as_posix(),
        "elapsed_seconds": elapsed,
    }

    meta_path = out_mask_path.parent / "vlm_selection.json"
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="AlphaGen VLM Grounding & Segmentation")
    parser.add_argument("--image", required=True, type=Path, help="Reference frame image path")
    parser.add_argument("--prompt", required=True, type=str, help="Natural language object description")
    parser.add_argument("--output", type=Path, default=None, help="Output mask PNG path")
    parser.add_argument("--sam2", type=Path, default=DEFAULT_SAM2_CHECKPOINT, help="SAM2 checkpoint path")
    parser.add_argument("--expand", type=float, default=1.2, help="Alpha expansion ratio (default: 1.2)")
    args = parser.parse_args()

    result = analyze_and_segment(args.image, args.prompt, args.output, args.sam2, expand_ratio=args.expand)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
