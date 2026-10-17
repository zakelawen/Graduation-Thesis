import gzip
import json
import os
import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
METRICS = ('recall_retrieval', 'f1_retrieval', 'recall_no_retrieval', 'f1_no_retrieval')

def read_jsonl(path):
    with gzip.open(os.path.join(ROOT, path), 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def read_json(path):
    with gzip.open(os.path.join(ROOT, path), 'rt', encoding='utf-8') as f:
        return json.load(f)

def class_scores(y_need, p_need):
    (y, p) = (np.asarray(y_need), np.asarray(p_need))
    (r1, f1) = precision_recall_fscore_support(y, p, average='binary', pos_label=1, zero_division=0)[1:3]
    (r0, f0) = precision_recall_fscore_support(y, p, average='binary', pos_label=0, zero_division=0)[1:3]
    return [round(100 * v, 2) for v in (r1, f1, r0, f0)]

def best_threshold(scores, labels, pos_label):
    (best_f1, best_t) = (-1.0, 0.0)
    for t in np.arange(0.01, 1.0, 0.01):
        f1 = precision_recall_fscore_support(labels, (scores >= t).astype(int), average='binary', pos_label=pos_label, zero_division=0)[2]
        if f1 > best_f1:
            (best_f1, best_t) = (f1, t)
    return best_t

def decision_row(row):
    data = read_jsonl(row['file'])
    y = [int(str(r[row['label_field']]) == row['label_needs_value']) for r in data]
    p = [int(str(r[row['pred_field']]) == row['pred_needs_value']) for r in data]
    return class_scores(y, p)

def classifier_row(row):
    model = joblib.load(os.path.join(ROOT, row['classifier']))
    (val, test) = (read_json(row['validation']), read_json(row['test']))
    s_val = model.predict_proba(np.array([d['normalized_truth_values'] for d in val]))[:, 1]
    y_val = np.array([d['label'] for d in val])
    s_test = model.predict_proba(np.array([d['normalized_truth_values'] for d in test]))[:, 1]
    y_need = (np.array([d['label'] for d in test]) == 0).astype(int)
    t_need = best_threshold(s_val, y_val, 0)
    return class_scores(y_need, (s_test < t_need).astype(int))

def evaluate():
    rows = json.load(open(os.path.join(ROOT, 'results', 'classifier', 'config.json'), encoding='utf-8'))
    out = {}
    for row in rows:
        values = decision_row(row) if row['type'] == 'decision' else classifier_row(row)
        out[row['dataset'], row['method']] = dict(zip(METRICS, values))
    return out

def main():
    print(f"{'dataset':8s} {'method':22s} " + ' '.join((f'{m:>20s}' for m in METRICS)))
    for ((dataset, method), values) in evaluate().items():
        print(f'{dataset:8s} {method:22s} ' + ' '.join((f'{values[m]:20.2f}' for m in METRICS)))
if __name__ == '__main__':
    main()
