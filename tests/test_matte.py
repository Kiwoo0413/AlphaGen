import unittest

import numpy as np

from alphagen_matte import Settings, fill_small_holes, refine


class MatteTests(unittest.TestCase):
    def test_empty_and_full_are_stable(self):
        for value in (0.0, 1.0):
            result = refine(np.full((32, 32), value, np.float32))
            np.testing.assert_allclose(result["final"], value, atol=1e-6)

    def test_background_gray_removed_core_locked_edge_soft(self):
        alpha = np.full((64, 64), 0.035, np.float32)
        alpha[16:48, 16:48] = 0.72
        alpha[15, 16:48] = 0.3
        result = refine(alpha, Settings(feather=0))
        self.assertEqual(float(result["final"][0, 0]), 0)
        self.assertEqual(float(result["final"][32, 32]), 1)
        self.assertGreater(float(result["final"][15, 30]), 0)
        self.assertLess(float(result["final"][15, 30]), 1)
        self.assertEqual(float(result["core"][16, 30]), 0)

    def test_fill_only_small_enclosed_holes(self):
        binary = np.ones((32, 32), np.uint8)
        binary[0:3, 0:3] = 0  # touches the outside: preserve
        binary[6:8, 6:8] = 0  # small enclosed hole: fill
        binary[16:24, 16:24] = 0  # real opening larger than limit: preserve
        filled = fill_small_holes(binary, 8)
        self.assertEqual(int(filled[1, 1]), 0)
        self.assertEqual(int(filled[6, 6]), 1)
        self.assertEqual(int(filled[20, 20]), 0)

    def test_hole_filling_is_optional(self):
        alpha = np.ones((32, 32), np.float32) * 0.7
        alpha[15, 15] = 0
        settings = dict(feather=0, core_inset=0, edge_restore=0)
        self.assertEqual(float(refine(alpha, Settings(**settings))["final"][15, 15]), 0)
        self.assertEqual(float(refine(alpha, Settings(fill_holes=True, **settings))["final"][15, 15]), 1)

    def test_zero_core_strength_disables_fill(self):
        alpha = np.ones((32, 32), np.float32) * 0.7
        output = refine(alpha, Settings(core_strength=0, feather=0, edge_restore=0))
        self.assertEqual(float(output["core"].max()), 0)
        self.assertLess(float(output["final"].max()), 1)

    def test_invalid_settings_and_nan_fail(self):
        for s in (Settings(black_clip=0.9, white_clip=0.8), Settings(gamma=0), Settings(core_inset=1.5)):
            with self.assertRaises(ValueError):
                refine(np.zeros((8, 8)), s)
        with self.assertRaises(ValueError):
            refine(np.full((8, 8), np.nan))

    def test_raw_input_not_modified(self):
        alpha = np.random.default_rng(4).random((64, 64), dtype=np.float32)
        before = alpha.copy()
        result = refine(alpha)
        np.testing.assert_array_equal(alpha, before)
        for matte in result.values():
            self.assertTrue(np.isfinite(matte).all())
            self.assertGreaterEqual(float(matte.min()), 0)
            self.assertLessEqual(float(matte.max()), 1)


if __name__ == "__main__":
    unittest.main()
