import unittest
from pathlib import Path

import numpy as np
import torch

from dataset import load_records, load_arrays
from experiment import augment, loss_fn, scores
from model import UNet
from preprocessing import METHODS, preprocess, srad


class ExperimentTests(unittest.TestCase):
    def test_srad_constant_and_invalid_input(self):
        for value in (0, 127, 255):
            image = np.full((17, 19), value, np.uint8)
            np.testing.assert_array_equal(srad(image), image)
        with self.assertRaises(ValueError):
            srad(np.zeros((4, 4), np.float32))
        with self.assertRaises(ValueError):
            srad(np.zeros((4, 4), np.uint8), step=1.1)

    def test_srad_smoothing_symmetry_and_mean(self):
        rng = np.random.default_rng(1)
        image = np.clip(128*rng.gamma(10, 0.1, (32, 32)), 1, 255).astype(np.uint8)
        filtered = srad(image)
        self.assertTrue(np.isfinite(filtered).all())
        self.assertLess(filtered.var(), image.var())
        self.assertLess(abs(float(filtered.mean()) - float(image.mean())), 0.1)
        np.testing.assert_allclose(srad(image[::-1, ::-1])[::-1, ::-1], filtered, atol=1)

    def test_median_and_combination_order(self):
        image = np.full((32, 32), 100, np.uint8)
        image[16, 16] = 255
        self.assertEqual(preprocess(image, "median")[16, 16], 100)
        np.testing.assert_array_equal(preprocess(image, "clahe_srad"), srad(preprocess(image, "clahe")))
        for method in METHODS:
            result = preprocess(image, method)
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.uint8)
        with self.assertRaises(ValueError):
            preprocess(image, "unknown")

    def test_metrics_empty_and_overlap(self):
        pred = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 0]])
        truth = torch.tensor([[0, 0], [1, 0], [1, 0], [1, 0]])
        dice, iou = scores(pred, truth)
        torch.testing.assert_close(dice, torch.tensor([1., 1., 2/3, 0.]))
        torch.testing.assert_close(iou, torch.tensor([1., 1., 0.5, 0.]))

    def test_unet_backward_and_synchronized_augmentation(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)
        masks = torch.zeros(2, 1, 64, 64)
        masks[:, :, 15:45, 20:40] = 1
        images, targets = augment(masks, masks)
        self.assertGreater(scores(images >= 0.5, targets)[0].min().item(), 0.98)
        model = UNet()
        logits = model(images)
        self.assertEqual(logits.shape, masks.shape)
        loss = loss_fn(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_real_manifest_has_no_group_or_exact_duplicate_leakage(self):
        root = Path("data/Dataset_BUSI_with_GT")
        if not root.exists():
            self.skipTest("Run prepare_data.py for real-data validation")
        records = load_records(root, Path("metadata/dataset_comment_list.csv"))
        self.assertEqual(len(records), 780)
        self.assertEqual(sum(r["clean"] for r in records), 529)
        for key in ("group", "sha256_pixels"):
            partition = {}
            for row in records:
                partition.setdefault(row[key], set()).add(row["split"])
            self.assertTrue(all(len(v) == 1 for v in partition.values()))
        for split in ("train", "val", "test"):
            self.assertEqual({r["category"] for r in records if r["clean"] and r["split"] == split},
                             {"benign", "malignant", "normal"})
        multifocal = next(r for r in records if len(r["masks"]) > 1)
        _, masks = load_arrays(root, [multifocal], 256)
        self.assertGreater(int(masks.sum()), 0)


if __name__ == "__main__":
    unittest.main()
