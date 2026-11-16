"""Pure helpers for a strict presence gate; scores are not probabilities."""
import math


def hard_negative_ids(manifest, train_rows):
    allowed = {r['image'] for r in train_rows if r['category'] == 'normal'}
    images = set(manifest['images'])
    if manifest.get('split') != 'train' or not images <= allowed:
        raise ValueError('Hard negatives must be normal images in the current training split')
    return images


def keep(score, threshold):
    return float(score) > float(threshold)


def calibrate(rows, split):
    if split != 'val' or any(r.get('split', split) != 'val' for r in rows):
        raise ValueError('Threshold fitting is permitted only on validation data')
    if any(not math.isfinite(float(r['presence_score'])) for r in rows):
        raise ValueError('Nonfinite score')
    negatives = [float(r['presence_score']) for r in rows if r['category'] == 'normal']
    if not negatives:
        raise ValueError('Calibration requires normal images')
    return max(negatives)


def metrics(rows):
    normal = [r for r in rows if r['category'] == 'normal']
    lesion = [r for r in rows if r['category'] != 'normal']
    mean = lambda values: sum(values)/len(values) if values else None
    fp = sum(r['pred_pixels'] > 0 for r in normal)
    return dict(n=len(rows), normal_n=len(normal), lesion_n=len(lesion), fp=fp, tn=len(normal)-fp,
                normal_fp_rate=fp/len(normal) if normal else None,
                dice=mean([r['dice'] for r in rows]), lesion_dice=mean([r['dice'] for r in lesion]),
                lesion_no_prediction_rate=mean([r['n_pred'] == 0 for r in lesion]),
                lesion_empty_rate=mean([r['pred_pixels'] == 0 for r in lesion]),
                lesion_zero_overlap_rate=mean([r['intersection'] == 0 for r in lesion]))
