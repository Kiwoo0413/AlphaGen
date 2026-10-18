import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from alphagen_worker import ROOT, bake, export_raw, job_path, prepare_input, read_alpha, save_alpha


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "jobs", prefix="test_")
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
            prepare_input(self.job, {"first": 1, "last": 10})

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


if __name__ == "__main__":
    unittest.main()
