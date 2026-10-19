import os
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
RAG_PROBE = os.environ.get('RAG_PROBE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rag_probe'))
RAG = Path(RAG_PROBE)
CLASS = RAG / 'Rebuttal/results/popqa1106_generalization/classification'
METHODS = [('MedCPT score', 'retriever_score.json', None), ('BGE Reranker', 'bge_reranker.json', CLASS / 'bge_reranker.json'), ('Utility CE', 'utility_cross_encoder.json', CLASS / 'utility_cross_encoder.json'), ('CRAG Evaluator', 'crag_evaluator.json', CLASS / 'crag_evaluator.json'), ('DUS', 'dus_utilityqwen.json', CLASS / 'dus/retrieved_utilityqwen_samples1_summary.json'), ('LLM Judge', 'llm_utility_judge.json', CLASS / 'llm_utility_judge.json'), ('Self-RAG-7B', 'selfrag7b_relevance.json', RAG / 'Rebuttal/results/selfrag7b_classification/retrieved.json'), ('SePer (Oracle) Qwen3-4B', 'seper_qwen3_4b.json', 'oracle'), ('SePer (Oracle) Gemma2-2B', 'seper_gemma2b.json', 'oracle'), ('Ours Qwen3-4B (zero-shot)', 'ours_qwen3_4b.json', 'ours'), ('Ours Gemma2-2B (zero-shot)', 'ours_gemma2b.json', 'ours'), ('Ours Qwen3-4B (medical)', 'ours_med_qwen3_4b.json', 'ours'), ('Ours Gemma2-2B (medical)', 'ours_med_gemma2b.json', 'ours')]

def threshold_for(spec):
    if spec is None:
        return (None, 'greater')
    if spec == 'ours':
        return (0.5, 'greater')
    if spec == 'oracle':
        return (1e-12, 'greater')
    obj = json.loads(Path(spec).read_text(encoding='utf-8'))
    return (float(obj['threshold']), str(obj.get('direction', 'greater')))

def top1_scores(path):
    data = json.load(open(path, encoding='utf-8'))['data']
    out = {}
    for row in data:
        top = next((d for d in row['scored_docs'] if int(d['rank']) == 0))
        out[f"{row['dataset']}:{row['id']}"] = (float(top['score']), (top.get('doc') or {}).get('id'))
    return out

def metrics(y, s, thr, direction):
    res = {'n': int(len(y)), 'pos': int(y.sum()), 'auroc': 100 * roc_auc_score(y, s if direction == 'greater' else -s), 'auprc': 100 * average_precision_score(y, s if direction == 'greater' else -s)}
    if thr is not None:
        p = (s >= thr).astype(int) if direction == 'greater' else (s <= thr).astype(int)
        res.update(acc=100 * float((p == y).mean()), macro_f1=100 * f1_score(y, p, average='macro'), balanced_acc=100 * balanced_accuracy_score(y, p))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--scores-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    labels = {}
    for line in open(args.labels, encoding='utf-8'):
        rec = json.loads(line)
        if rec.get('status') == 'ok' and rec.get('label_name') in ('relevant', 'distracting'):
            labels[rec['id']] = rec
    ids = sorted(labels)
    y_all = np.array([int(labels[i]['label_name'] == 'relevant') for i in ids])
    ds_all = np.array([labels[i]['dataset'] for i in ids])
    rows = []
    for (name, fname, spec) in METHODS:
        path = Path(args.scores_dir) / fname
        if not path.exists():
            continue
        scored = top1_scores(path)
        stale = [i for i in ids if i not in scored or (scored[i][1] is not None and scored[i][1] != labels[i]['doc_id'])]
        if stale:
            raise SystemExit(f'{fname}: {len(stale)} labeled items missing or with a different top-1 passage')
        s_all = np.array([scored[i][0] for i in ids])
        (thr, direction) = threshold_for(spec)
        for ds in ['ALL'] + sorted(set(ds_all)):
            mask = np.ones(len(ids), bool) if ds == 'ALL' else ds_all == ds
            if len(set(y_all[mask])) < 2:
                continue
            rows.append({'method': name, 'dataset': ds, 'threshold': thr, **metrics(y_all[mask], s_all[mask], thr, direction)})
    cols = ['method', 'dataset', 'n', 'pos', 'auroc', 'auprc', 'acc', 'macro_f1', 'balanced_acc', 'threshold']
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\t'.join(cols) + '\n')
        for r in rows:
            f.write('\t'.join(('' if r.get(c) is None else f'{r[c]:.2f}' if isinstance(r[c], float) and c != 'threshold' else str(r[c]) for c in cols)) + '\n')
    for r in rows:
        if r['dataset'] == 'ALL':
            acc = f"{r['acc']:.2f}" if 'acc' in r else '  -  '
            print(f"{r['method']:26s} AUROC {r['auroc']:.2f}  Acc {acc}  (n={r['n']}, relevant={r['pos']})")
if __name__ == '__main__':
    main()
