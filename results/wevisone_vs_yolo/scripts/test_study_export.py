"""Check the exported narrative against the actual generated study artifacts."""
import json
import unittest

import analyze as A
from sync_study import readme


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = json.loads((A.OUT / "report.json").read_text(encoding="utf-8"))
        cls.rep.setdefault("low_data_status", json.loads((A.OUT / "low_data_status.json").read_text()))
        cls.strict = json.loads((A.OUT / "strict_f1.json").read_text())
        cls.hits = json.loads((A.OUT / "common_hits.json").read_text())
        cls.original = (A.common.NTUT / "results/wevisone_vs_yolo/README.md").read_text(encoding="utf-8")

    def test_no_decoder_causality_overclaim_remains(self):
        output = readme(self.original, self.rep, self.strict, self.hits)
        self.assertNotIn("不是 decoder", output)
        self.assertIn("不可引用固定門檻", output)
        self.assertIn("IoU ≥ 0.50", output)

    def test_repeated_export_is_idempotent(self):
        first = readme(self.original, self.rep, self.strict, self.hits)
        self.assertEqual(first, readme(first, self.rep, self.strict, self.hits))

    def test_artifact_counts_are_consistent(self):
        for spec in self.strict.values():
            for model in spec.values():
                for ops in model.values():
                    self.assertGreaterEqual(ops["any_overlap"]["tp"], ops["iou_0.25"]["tp"])
                    self.assertGreaterEqual(ops["iou_0.25"]["tp"], ops["iou_0.50"]["tp"])
                    for r in ops.values():
                        self.assertAlmostEqual(r["f1"], 2*r["tp"] / (2*r["tp"]+r["fp"]+r["fn"]))
        for spec in self.hits.values():
            for ops in spec.values():
                for r in ops.values():
                    self.assertEqual(r["both"]+r["only_a"]+r["only_b"]+r["neither"], r["lesion_n"])
                    self.assertEqual(sum(len(s["ids"]) for s in r["per_seed"]), r["both"])


if __name__ == "__main__":
    unittest.main()
