import csv
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import torch
from followup import run, matched_training_indices, summarize


class FollowupTests(unittest.TestCase):
    def test_export_preserves_optional_reuse_provenance(self):
        config=dict(cohorts=['matched','cleaned'],methods=['raw'],seeds=[42,43,44],budgets=[2,4])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for cohort in config['cohorts']:
                for seed in config['seeds']:
                    for budget in config['budgets']:
                        folder=root/'checkpoints'/f'{cohort}_raw_{seed}_steps{budget}'
                        folder.mkdir(parents=True)
                        row=dict(cohort=cohort,method='raw',seed=seed,updates=budget,val_dice=.5)
                        if cohort=='cleaned':
                            row['reused_from']='source'
                        (folder/'metrics.json').write_text(json.dumps(row))
            summarize(root,config)
            with (root/'runs.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),12)
            self.assertTrue(all(r['reused_from']=='source' for r in rows if r['cohort']=='cleaned'))

    def test_completed_matched_experiment_integrity(self):
        root = Path('results/quality_matched')
        if not (root/'runs.csv').exists():
            self.skipTest('Matched experiment is not complete')
        config=json.loads((root/'config.json').read_text())
        self.assertFalse(config['test_evaluated'])
        with (root/'sampled_original.csv').open() as f:
            sample=list(csv.DictReader(f))
        self.assertEqual(len(sample),368)
        self.assertTrue(all(r['split']=='train' for r in sample))
        self.assertEqual(Counter(r['category'] for r in sample),{'benign':207,'malignant':112,'normal':49})
        with (root/'manifest.csv').open() as f:
            manifest=list(csv.DictReader(f))
        self.assertNotIn('test',{r['split'] for r in manifest})
        groups={}
        for row in manifest:
            groups.setdefault(row['group'],set()).add(row['split'])
        self.assertTrue(all(len(s)==1 for s in groups.values()))
        with (root/'runs.csv').open() as f:
            runs=list(csv.DictReader(f))
        expected={(c,m,str(s),str(b)) for c in config['cohorts'] for m in config['methods']
                  for s in config['seeds'] for b in config['budgets']}
        self.assertEqual(len(runs),len(expected))
        self.assertEqual({(r['cohort'],r['method'],r['seed'],r['updates']) for r in runs},expected)
        for row in runs:
            self.assertEqual(int(row['train_n']),368)
            self.assertEqual(int(row['val_n']),85)
            folder=root/'checkpoints'/f"{row['cohort']}_{row['method']}_{row['seed']}_steps{row['updates']}"
            with (folder/'history.csv').open() as f:
                history=list(csv.DictReader(f))
            self.assertEqual([int(r['step']) for r in history],list(range(35,int(row['updates'])+1,35)))
            self.assertEqual(float(row['val_dice']),max(float(r['val_dice']) for r in history))
            self.assertTrue(0 <= float(row['val_dice']) <= 1)

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
