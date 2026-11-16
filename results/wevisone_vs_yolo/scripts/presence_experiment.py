"""Measure a first-divergent-token gate; calibrate on val, then evaluate test.

The gate only suppresses generated masks. It cannot recover a missed lesion.
All artifacts are experimental and written to a new directory.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import common
from evaluate import apply_finetune, predict_union
from fpr_metrics import calibrate, keep, metrics


@torch.inference_mode()
def presence_score(engine, image):
    inputs = engine._inputs(engine._messages(image, common.PROMPT), image, generation=True)
    # The trained alternatives start with different tokens: None versus <ref>.
    negative = engine.tokenizer.encode('None', add_special_tokens=False)[0]
    positive = engine.tokenizer.encode('<ref>breast lesion</ref>', add_special_tokens=False)[0]
    if negative == positive:
        raise ValueError('Response alternatives must have different first tokens')
    out = engine.model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1)
    logits = out.logits[0, -1].float()
    lp = logits.log_softmax(-1)
    return dict(presence_score=float(logits[positive]-logits[negative]),
                lesion_first_token_logprob=float(lp[positive]), none_first_token_logprob=float(lp[negative]))


def collect(engine, split, out, normals_only=False):
    records = common.clean_split(split)
    if normals_only:
        if split != 'train':
            raise ValueError('Hard negative mining requires train split')
        records = [r for r in records if r['category'] == 'normal']
    result, probs, preds = [], {}, {}
    for index, row in enumerate(records):
        image, instances, union = common.load_sample(row)
        score = presence_score(engine, image)
        response, prob, n = predict_union(engine, image, instances, 'generate')
        pred, gt = common.to_eval_res(prob, union)
        dice, iou = common.scores(pred, gt)
        key = row['image'].replace('/', '__')
        probs[key] = np.zeros((256,256), np.float32) if prob is None else cv2.resize(prob.astype(np.float32), (256,256))
        preds[key] = pred
        result.append(dict(image=row['image'], category=row['category'], split=split, **score,
                           response=response, n_pred=n, pred_pixels=int(pred.sum()),
                           intersection=int((pred.astype(bool)&gt.astype(bool)).sum()), dice=dice, iou=iou))
        if (index+1)%10 == 0: print(f'{split} {index+1}/{len(records)}', flush=True)
    write_csv(out/f'{split}_scores.csv', result)
    np.savez_compressed(out/f'{split}_prob256.npz', **probs)
    np.savez_compressed(out/f'{split}_pred256.npz', **preds)
    return result


def write_csv(path, rows):
    if not rows: return
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def gated(rows, threshold):
    return [dict(r) if keep(r['presence_score'], threshold) else
            dict(r, n_pred=0, pred_pixels=0, intersection=0, dice=float(r['category']=='normal'), iou=float(r['category']=='normal')) for r in rows]


def save_gated_masks(out, split, rows, threshold):
    with np.load(out/f'{split}_pred256.npz') as masks:
        np.savez_compressed(out/f'{split}_gated_pred256.npz', **{
            r['image'].replace('/', '__'): masks[r['image'].replace('/', '__')] if keep(r['presence_score'], threshold)
            else np.zeros_like(masks[r['image'].replace('/', '__')]) for r in rows})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lora', required=True); ap.add_argument('--decoder', required=True)
    ap.add_argument('--out', required=True)
    args=ap.parse_args(); out=Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'args.json').write_text(json.dumps(vars(args), indent=2))
    engine=common.load_engine(); apply_finetune(engine,args.lora,args.decoder)
    val=collect(engine,'val',out)
    threshold=calibrate(val,'val')
    calibration=dict(threshold=threshold, comparison='score > threshold', split='val', normal_n=sum(r['category']=='normal' for r in val),
                     score='logit(first lesion token) - logit(first None token)',
                     warning='Relative score, not calibrated probability. Zero observed val FP is not a population guarantee.')
    (out/'calibration.json').write_text(json.dumps(calibration,indent=2))
    report=dict(val_raw=metrics(val), val_gated=metrics(gated(val,threshold)))
    write_csv(out/'val_gated.csv', gated(val,threshold))
    save_gated_masks(out,'val',val,threshold)
    # Threshold is persisted before reading test labels or scores.
    test=collect(engine,'test',out)
    report.update(test_raw=metrics(test),test_gated=metrics(gated(test,threshold)))
    write_csv(out/'test_gated.csv', gated(test,threshold))
    save_gated_masks(out,'test',test,threshold)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    train=collect(engine,'train',out,normals_only=True)
    hard=[r for r in train if r['pred_pixels']>0]
    manifest=dict(split='train', source_lora=str(Path(args.lora).resolve()),
                  images=[r['image'] for r in hard], records=hard)
    (out/'hard_negatives.json').write_text(json.dumps(manifest,indent=2))
    report['train_normal_fp']=metrics(train)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
