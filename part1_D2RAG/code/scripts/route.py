import argparse
import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d2rag.routing import needs_retrieval, select_threshold
from d2rag.utils import load_json_or_jsonl, load_jsonl, question_key, write_jsonl

def truth_values(rec):
    if rec.get('normalized_truth_values') is not None:
        return rec['normalized_truth_values']
    return (rec.get('output') or {}).get('normalized_truth_values')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['llama', 'qwen'])
    ap.add_argument('--stage1', required=True)
    ap.add_argument('--seper', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--settings', default=os.path.join(os.path.dirname(__file__), '..', 'configs', 'settings.json'))
    ap.add_argument('--data-root', default='.')
    args = ap.parse_args()
    import joblib
    cfg = json.load(open(args.settings, encoding='utf-8'))['classifier'][args.model]
    clf = joblib.load(os.path.join(args.data_root, cfg['model']))
    val = load_json_or_jsonl(os.path.join(args.data_root, cfg['validation']))
    threshold = select_threshold(clf, np.array([v['normalized_truth_values'] for v in val]), [v['label'] for v in val])
    scores = {}
    for rec in load_json_or_jsonl(args.stage1):
        tv = truth_values(rec)
        if tv is not None:
            scores[question_key(rec['question'])] = float(clf.predict_proba(np.array([tv]))[0, 1])
    (routed, missing) = ([], 0)
    for item in load_jsonl(args.seper):
        score = scores.get(question_key(item['question']))
        if score is None:
            missing += 1
            (item['needs_retrieval'], item['clf_score']) = (1, None)
        else:
            (item['needs_retrieval'], item['clf_score']) = (needs_retrieval(score, threshold), score)
        item['clf_threshold'] = threshold
        routed.append(item)
    write_jsonl(args.out, routed)
    n_ret = sum((r['needs_retrieval'] for r in routed))
    print(f'threshold={threshold:.2f}  retrieve {n_ret}/{len(routed)}  no stage-1 score: {missing}  -> {args.out}')
if __name__ == '__main__':
    main()
