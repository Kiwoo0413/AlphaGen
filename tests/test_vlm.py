import json
from pathlib import Path
import unittest
import numpy as np

from alphagen_vlm import (
    segment_with_sam2,
    ground_object_bbox_vlm,
    expand_mask,
    DEFAULT_SAM2_CHECKPOINT,
)


class VLMModuleTests(unittest.TestCase):
    def test_coordinate_scaling(self):
        w, h = 1920, 1080
        box_2d_1000 = [160, 265, 1000, 714]  # ymin, xmin, ymax, xmax
        ymin, xmin, ymax, xmax = box_2d_1000
        box_xyxy = [
            int(xmin / 1000 * w),
            int(ymin / 1000 * h),
            int(xmax / 1000 * w),
            int(ymax / 1000 * h),
        ]
        self.assertEqual(box_xyxy, [508, 172, 1370, 1080])

    def test_expand_mask_1_2x(self):
        import cv2
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 80:120] = 255  # w=40, h=100
        expanded = expand_mask(mask, expand_ratio=1.2)
        pts = cv2.findNonZero(expanded)
        x, y, w, h = cv2.boundingRect(pts)
        self.assertEqual(w, 48)   # 40 * 1.2 = 48
        self.assertEqual(h, 120)  # 100 * 1.2 = 120
        self.assertGreater(cv2.countNonZero(expanded), cv2.countNonZero(mask))

    def test_expand_mask_no_op(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        unmodified = expand_mask(mask, expand_ratio=1.0)
        np.testing.assert_array_equal(mask, unmodified)

    def test_expand_mask_empty(self):
        empty = np.zeros((40, 40), dtype=np.uint8)
        result = expand_mask(empty, expand_ratio=1.2)
        np.testing.assert_array_equal(empty, result)

    def test_expand_mask_float(self):
        import cv2
        fmask = np.zeros((100, 100), dtype=np.float32)
        fmask[20:60, 30:70] = 1.0  # w=40, h=40
        expanded = expand_mask(fmask, expand_ratio=1.2)
        pts = cv2.findNonZero((expanded > 0.01).astype(np.uint8))
        x, y, w, h = cv2.boundingRect(pts)
        self.assertEqual(w, 48)  # 40 * 1.2 = 48
        self.assertEqual(h, 48)  # 40 * 1.2 = 48

    def test_sam2_inference_headless(self):
        try:
            from transformers import Sam2Model, Sam2Processor
        except ImportError:
            self.skipTest("Sam2Model not supported in this Python environment")
        if not DEFAULT_SAM2_CHECKPOINT.is_dir():
            self.skipTest("SAM2 checkpoint not available locally")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_img[20:80, 20:80] = 255
        box_2d = [200, 200, 800, 800]
        mask, iou = segment_with_sam2(dummy_img, box_2d)
        self.assertEqual(mask.shape, (100, 100))
        self.assertGreater(mask.sum(), 0)


if __name__ == "__main__":
    unittest.main()
