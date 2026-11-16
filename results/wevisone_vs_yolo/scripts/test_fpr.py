import unittest
import fpr_metrics as f

class MetricsTests(unittest.TestCase):
    def test_gate_rejects_tied_normal_scores(self):
        rows = [dict(category='normal', presence_score=2.), dict(category='normal', presence_score=1.), dict(category='benign', presence_score=3.)]
        self.assertEqual(f.calibrate(rows, 'val'), 2.)
        self.assertFalse(f.keep(2., 2.))
        self.assertTrue(f.keep(3., 2.))

    def test_calibration_rejects_leakage_and_invalid_inputs(self):
        for split, rows in [('test', [dict(category='normal', presence_score=1)]), ('val', [dict(category='normal', presence_score=1, split='test')]), ('val', []), ('val', [dict(category='normal', presence_score=float('nan'))])]:
            with self.assertRaises(ValueError): f.calibrate(rows, split)

    def test_metrics_distinguish_wrong_location_from_empty(self):
        rows = [dict(category='normal', dice=0., pred_pixels=1, intersection=0, n_pred=1),
                dict(category='normal', dice=1., pred_pixels=0, intersection=0, n_pred=0),
                dict(category='benign', dice=0., pred_pixels=10, intersection=0, n_pred=1),
                dict(category='malignant', dice=0., pred_pixels=0, intersection=0, n_pred=0)]
        m=f.metrics(rows)
        self.assertEqual((m['fp'],m['tn']), (1,1))
        self.assertEqual(m['lesion_empty_rate'], .5)
        self.assertEqual(m['lesion_zero_overlap_rate'], 1.)

    def test_shared_evaluator_reports_actual_misses(self):
        from evaluate import summarize
        row=dict(category='benign', dice=0., iou=0., native_dice=0., n_pred=1,
                 pred_pixels=10, intersection=0)
        m=summarize([row])
        self.assertEqual(m['lesion_zero_overlap_rate'], 1.)
        self.assertEqual(m['lesion_empty_rate'], 0.)

    def test_hard_negative_weights_reject_test_and_lesion_rows(self):
        rows=[dict(image='n',category='normal'),dict(image='b',category='benign')]
        self.assertEqual(f.hard_negative_ids(dict(split='train',images=['n']),rows), {'n'})
        for manifest in [dict(split='test',images=['n']),dict(split='train',images=['b']),dict(split='train',images=['unknown'])]:
            with self.assertRaises(ValueError): f.hard_negative_ids(manifest,rows)

if __name__ == '__main__': unittest.main()
