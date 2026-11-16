"""Small synthetic checks for the localization analysis contract."""
import unittest
import warnings
import math

from f1_table import counts, common_hits
from analyze import at_fpr, main_conclusions, paired_bootstrap


def row(ident, category="benign", iou=0.5, pred=10, inter=5, dice=2/3):
    return dict(id=ident, patient=ident, category=category, iou=iou,
                pred_pixels=pred, intersection=inter, dice=dice)


class LocalizationTests(unittest.TestCase):
    def test_strict_boundary_wrong_and_empty_predictions(self):
        rows = [row("hit"), row("poor", iou=0.49),
                row("wrong", iou=0, inter=0), row("empty", iou=0, pred=0, inter=0),
                row("normal_empty", "normal", iou=1, pred=0, inter=0),
                row("normal_fp", "normal", iou=0, inter=0)]
        self.assertEqual(counts(rows, min_iou=0.5), (1, 3, 3))
        self.assertEqual(counts(rows), (2, 2, 2))

    def test_common_hits_aligns_ids_and_reports_all_partitions(self):
        a = [row("both", dice=0.8), row("a"), row("b", iou=0.1), row("neither", iou=0)]
        b = [row("neither", iou=0), row("b"), row("a", iou=0.1), row("both", dice=0.7)]
        result = common_hits(a, b, min_iou=0.5)
        self.assertEqual([result[k] for k in ("both", "only_a", "only_b", "neither")], [1, 1, 1, 1])
        self.assertEqual(result["ids"], ["both"])
        self.assertAlmostEqual(result["dice_diff"], 0.1)

    def test_no_common_hit_is_undefined(self):
        result = common_hits([row("x", iou=0)], [row("x", iou=0)], min_iou=0.5)
        self.assertIsNone(result["dice_diff"])

    def test_mismatched_cohorts_fail(self):
        with self.assertRaises(ValueError):
            common_hits([row("x")], [row("y")])

    def test_fpr_budget_is_not_exact_matching(self):
        pts = [(0.12, 0.9, 0.1, 1), (0.08, 0.7, 0.2, 2), (0.04, 0.6, 0.3, 3)]
        self.assertEqual(at_fpr(pts, 0.1), pts[1])

    def test_conclusion_does_not_transfer_fixed_ci_to_matched_fpr(self):
        result = main_conclusions(dict(paired_bootstrap={"uclm:test | WeVisOne gate vs YOLO cal": dict(
            fpr_diff=0.057, fpr_diff_ci=[-0.003, 0.125], lesion_dice_diff=0.202,
            lesion_dice_diff_ci=[0.116, 0.313])}))
        self.assertIn("區間跨過 0", result)
        self.assertIn("不可引用固定門檻", result)

    def test_summary_accepts_json_threshold_keys(self):
        point = dict(lesion_dice=[0.7, 0.02, 3], fpr=[0.09, 0.01, 3])
        result = main_conclusions(dict(paired_bootstrap={}, matched_fpr={"uclm:test": {
            "WeVisOne (presence gate)": {"0.1": point}, "YOLO (conf threshold)": {"0.1": point}}}))
        self.assertIn("FPR ≤ 10%", result)

    def test_bootstrap_without_normals_is_undefined_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = paired_bootstrap([[row("x")]], [[row("x")]], n_boot=20)
        self.assertTrue(math.isnan(result["fpr_diff_ci"][0]))
        self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()
