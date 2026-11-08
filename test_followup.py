import json
from pathlib import Path
import tempfile
import unittest

import torch
from followup import run, matched_training_indices


class FollowupTests(unittest.TestCase):
    def test_sampling_matches_classes_without_holdout_or_replacement(self):
        records = [dict(split=split, category=category, clean=i<3)
                   for split in ('train','val','test')
                   for category in ('benign','malignant','normal') for i in range(5)]
        selected = matched_training_indices(records, 2026)
        self.assertEqual(selected, matched_training_indices(records, 2026))
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(len(selected), 9)
        self.assertTrue(all(records[i]['split'] == 'train' for i in selected))
        for category in ('benign','malignant','normal'):
            self.assertEqual(sum(records[i]['category'] == category for i in selected), 3)

    def test_exact_budgets_and_best_validation_snapshots(self):
        torch.set_num_threads(2)
        images = torch.rand(6, 1, 32, 32)
        masks = (images > .7).float()
        config = dict(lr=.001, budgets=[2,4], batch_size=2, validate_every=2)
        with tempfile.TemporaryDirectory() as temp:
            prefix = Path(temp) / 'original_raw_42'
            run(images, masks, torch.tensor([0,1,2,3]), torch.tensor([4,5]), config, 42, prefix)
            scores = []
            for budget in config['budgets']:
                folder = Path(temp) / f'original_raw_42_steps{budget}'
                metrics = json.loads((folder / 'metrics.json').read_text())
                self.assertEqual(metrics['updates'], budget)
                self.assertLessEqual(metrics['best_step'], budget)
                state = torch.load(folder / 'best.pt', weights_only=True)
                self.assertEqual(state['val_dice'], metrics['val_dice'])
                self.assertEqual(metrics['val_n'], 2)
                scores.append(metrics['val_dice'])
            self.assertGreaterEqual(scores[1], scores[0])


if __name__ == '__main__':
    unittest.main()
