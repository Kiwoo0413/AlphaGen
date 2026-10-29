import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from alphagen_worker import ROOT, bake, export_raw, job_path, main, prepare_input, read_alpha, save_alpha


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test_")
        self.job = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_workspace_boundary(self):
        self.assertEqual(job_path(self.job), self.job.resolve())
        with self.assertRaises(ValueError):
            job_path(ROOT.parent)

    def test_roundtrip_16bit_and_frame_mapping(self):
        alpha = np.stack([np.full((8, 12), value, np.float32) for value in (0.1, 0.3, 0.9)])
        meta = export_raw(self.job, alpha, 1001, content_size=[10, 6], output_size=[20, 12])
        self.assertEqual((meta["first"], meta["last"]), (1001, 1003))
        self.assertEqual(meta["size"], [20, 12])
        for frame, expected in zip(range(1001, 1004), (0.1, 0.3, 0.9)):
            decoded = read_alpha(meta["raw_pattern"] % frame)
            np.testing.assert_allclose(decoded, expected, atol=1/65535)
        bake(self.job, meta, {})
        result = json.loads((self.job / "result.json").read_text())
        self.assertEqual(len(result["metrics"]), 3)
        self.assertEqual(json.loads((self.job / "status.json").read_text())["state"], "complete")

    def test_prepare_preserves_frame_range_and_mask_location(self):
        for frame in range(1001, 1010):
            source = np.full((40, 80, 3), frame - 1000, np.uint8)
            cv2.imwrite(str(self.job / f"source.{frame:06d}.png"), source)
            mask = np.zeros((40, 80), np.float32)
            mask[10:30, 20:60] = 1
            save_alpha(self.job / f"mask.{frame:06d}.png", mask)
        request = {"first": 1001, "last": 1009, "fps": 24, "max_side": 80,
                   "source_pattern": str(self.job / "source.%06d.png"),
                   "mask_pattern": str(self.job / "mask.%06d.png")}
        manifest = prepare_input(self.job, request)
        self.assertEqual(manifest["prepared_size"], [96, 64])
        self.assertEqual(manifest["resized_content_size"], [80, 40])
        cap = cv2.VideoCapture(str(self.job / "attention_mask.avi"))
        ok, frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        self.assertEqual(int(frame[20, 40, 0]), 255)
        self.assertEqual(int(frame[60, 90, 0]), 0)

    def test_reject_invalid_frame_count_before_export(self):
        with self.assertRaises(ValueError):
            prepare_input(self.job, {"first": 10, "last": 5})

    def test_prepare_input_pads_arbitrary_frames(self):
        # 10 frames (1001..1010) should be padded to 17 frames (next 8n+1)
        for frame in range(1001, 1011):
            source = np.full((40, 80, 3), 128, np.uint8)
            cv2.imwrite(str(self.job / f"source.{frame:06d}.png"), source)
            mask = np.zeros((40, 80), np.float32)
            mask[10:30, 20:60] = 1
            save_alpha(self.job / f"mask.{frame:06d}.png", mask)
        request = {"first": 1001, "last": 1010, "fps": 24, "max_side": 80,
                   "source_pattern": str(self.job / "source.%06d.png"),
                   "mask_pattern": str(self.job / "mask.%06d.png")}
        manifest = prepare_input(self.job, request)
        self.assertEqual(manifest["requested_frames"], 10)
        self.assertEqual(manifest["frames"], 17)

    def test_cancel_before_any_alpha_write(self):
        (self.job / "cancel.request").write_text("cancel")
        with self.assertRaises(InterruptedError):
            export_raw(self.job, np.zeros((1, 8, 8)), 1)
        self.assertEqual(list((self.job / "raw").glob("*.png")), [])

    def test_export_and_bake_to_custom_source_alpha_dir(self):
        custom_alpha_dir = self.job / "source_shot" / "alpha"
        alpha = np.stack([np.full((8, 12), 0.8, np.float32)])
        meta = export_raw(self.job, alpha, 1001, output_dir=custom_alpha_dir)
        self.assertTrue((custom_alpha_dir / "raw/alpha.001001.png").is_file())
        bake(self.job, meta, {}, output_dir=custom_alpha_dir)
        self.assertTrue((custom_alpha_dir / "final/alpha.001001.png").is_file())
        self.assertTrue((custom_alpha_dir / "raw/alpha.001001.png").is_file())
        self.assertFalse((custom_alpha_dir / "core").exists())
        self.assertFalse((custom_alpha_dir / "edge").exists())
        self.assertTrue((custom_alpha_dir / "result.json").is_file())
        res = json.loads((custom_alpha_dir / "result.json").read_text())
        self.assertIn("source_shot/alpha/final", res["final_pattern"].replace("\\", "/"))


    def test_cancel_pid_and_status(self):
        import os
        from alphagen_nuke import _is_pid_running
        self.assertTrue(_is_pid_running(os.getpid()))
        self.assertFalse(_is_pid_running(99999999))

    def test_cache_directory_resolution(self):
        import os
        from alphagen_nuke import get_nuke_cache_dir
        from alphagen_worker import get_jobs_root

        # 1. Default fallback is inside temp / nuke / AlphaGen
        cache = get_nuke_cache_dir()
        self.assertTrue(cache.is_dir())
        self.assertEqual(cache.name, "AlphaGen")

        # 2. ALPHAGEN_CACHE_DIR override
        custom = self.job / "custom_cache"
        os.environ["ALPHAGEN_CACHE_DIR"] = str(custom)
        try:
            self.assertEqual(get_nuke_cache_dir(), custom.resolve())
            self.assertEqual(get_jobs_root(), custom.resolve())
        finally:
            os.environ.pop("ALPHAGEN_CACHE_DIR", None)

    def test_sync_source_range_allows_over_121_frames(self):
        from alphagen_nuke import sync_source_range

        class MockKnob:
            def __init__(self, val=0):
                self.val = val
            def setValue(self, v):
                self.val = v
            def value(self):
                return self.val

        class MockSource:
            def __init__(self, first=1, last=240):
                self._knobs = {"first": MockKnob(first), "last": MockKnob(last)}
            def knobs(self):
                return self._knobs
            def __getitem__(self, key):
                return self._knobs[key]

        class MockGizmo:
            def __init__(self, src):
                self._src = src
                self._knobs = {
                    "first": MockKnob(1),
                    "last": MockKnob(1),
                    "ag_ref_frame": MockKnob(1),
                    "ag_status_label": MockKnob(""),
                }
            def knobs(self):
                return self._knobs
            def __getitem__(self, key):
                return self._knobs[key]
            def input(self, idx):
                return self._src if idx == 0 else None

        source = MockSource(first=1, last=240)
        gizmo = MockGizmo(source)
        sync_source_range(gizmo)
        self.assertEqual(gizmo["first"].value(), 1)
        self.assertEqual(gizmo["last"].value(), 240)
        self.assertGreater(gizmo["last"].value(), 121)

    def test_main_refine_mode_preserves_vlm_mask_path(self):
        import sys
        alpha = np.stack([np.full((8, 12), 0.7, np.float32)])
        meta = export_raw(self.job, alpha, 1001)
        meta["vlm_mask_path"] = str(self.job / "vlm_initial_mask.png")
        orig_res = self.job / "result.json"
        orig_res.write_text(json.dumps(meta), encoding="utf-8")
        req_file = self.job / "request.json"
        req_file.write_text(json.dumps({
            "mode": "refine",
            "result": str(orig_res),
            "output_dir": str(self.job)
        }), encoding="utf-8")

        orig_argv = sys.argv
        sys.argv = ["alphagen_worker.py", str(req_file)]
        try:
            main()
        finally:
            sys.argv = orig_argv

        res = json.loads(orig_res.read_text())
        self.assertEqual(res.get("vlm_mask_path"), str(self.job / "vlm_initial_mask.png"))
        self.assertIn("final_pattern", res)

    def test_roi_crop_and_stitch_small_target(self):
        # Small target in bottom-right of a 200x400 frame
        source = np.full((200, 400, 3), 100, np.uint8)
        cv2.imwrite(str(self.job / "source.000136.png"), source)
        mask = np.zeros((200, 400), np.float32)
        mask[140:180, 260:320] = 1.0
        mask_path = self.job / "vlm_initial_mask.png"
        save_alpha(mask_path, mask)

        request = {
            "first": 136,
            "last": 136,
            "fps": 24,
            "max_side": 128,
            "source_pattern": str(self.job / "source.%06d.png"),
            "mask_pattern": str(mask_path),
        }
        manifest = prepare_input(self.job, request)
        # Crop box around [260..320, 140..180] with 50% padding -> [230, 120, 350, 200] (120x80)
        self.assertEqual(manifest["crop_box"], [230, 120, 350, 200])
        self.assertEqual(manifest["source_size"], [400, 200])

        # Simulate LTX returning a full-white alpha inside the prepared canvas
        pw, ph = manifest["prepared_size"]
        fake_ltx_alpha = np.ones((1, ph, pw), dtype=np.float32)
        meta = export_raw(
            self.job,
            fake_ltx_alpha,
            136,
            manifest["resized_content_size"],
            manifest["source_size"],
            crop_box=manifest["crop_box"],
            mask_pattern=str(mask_path),
        )
        stitched = read_alpha(meta["raw_pattern"] % 136)
        self.assertEqual(stitched.shape, (200, 400))
        # Target center [160, 290] must be 1.0
        self.assertGreater(float(stitched[160, 290]), 0.99)
        # Far background [50, 50] must be strictly 0.0
        self.assertEqual(float(stitched[50, 50]), 0.0)


if __name__ == "__main__":
    unittest.main()
