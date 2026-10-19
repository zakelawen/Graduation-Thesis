import argparse
import glob
import json
import os
import re
from collections import defaultdict
LETTER = re.compile('^\\s*(?:(?:the\\s+)?(?:correct\\s+)?(?:answer|option)\\s*(?:is)?\\s*[:：\\-–—]?\\s*)?\\(?([A-E])(?:[).:\\s]|$)', re.I)
ANSWER_IS = re.compile('(?:answer\\s+is|option)\\s*[:：\\-–—]?\\s*\\(?([A-E])\\b', re.I)

def norm(s):
    return re.sub('[^a-z0-9 ]', ' ', (s or '').lower()).split()

def predict_letter(pred, options):
    pred = (pred or '').strip()
    m = LETTER.match(pred) or ANSWER_IS.search(pred)
    if m and m.group(1).upper() in options:
        return m.group(1).upper()
    p = ' '.join(norm(pred))
    if not p:
        return None
    hits = [k for (k, v) in options.items() if ' '.join(norm(v)) and (' '.join(norm(v)) == p or ' '.join(norm(v)) in p)]
    if len(hits) == 1:
        return hits[0]
    starts = [k for (k, v) in options.items() if ' '.join(norm(v)).startswith(p)]
    return starts[0] if len(starts) == 1 else None

def evaluate(path, questions):
    obj = json.load(open(path, encoding='utf-8'))
    by_ds = defaultdict(lambda : [0, 0, 0])
    for out in obj['outputs']:
        q = questions[int(out['idx'])]
        pred = out['generation']['prediction']
        letter = predict_letter(pred, q['options'])
        s = by_ds[q['dataset']]
        s[0] += 1
        s[1] += letter == q['answer']
        s[2] += letter is None
    meta = obj['meta']
    rows = []
    for (ds, (n, c, invalid)) in sorted(by_ds.items()):
        rows.append({'model': meta['model'], 'selector': meta['selector'], 'dataset': ds, 'n': n, 'acc': 100 * c / n, 'invalid': invalid, 'avg_ctx': meta['avg_selected_docs']})
    n = sum((v[0] for v in by_ds.values()))
    c = sum((v[1] for v in by_ds.values()))
    invalid = sum((v[2] for v in by_ds.values()))
    rows.append({'model': meta['model'], 'selector': meta['selector'], 'dataset': 'ALL', 'n': n, 'acc': 100 * c / n, 'invalid': invalid, 'avg_ctx': meta['avg_selected_docs']})
    rows.append({'model': meta['model'], 'selector': meta['selector'], 'dataset': 'MACRO', 'n': n, 'acc': sum((100 * v[1] / v[0] for v in by_ds.values())) / len(by_ds), 'invalid': invalid, 'avg_ctx': meta['avg_selected_docs']})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrieval', required=True)
    ap.add_argument('--glob', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    data = json.load(open(args.retrieval, encoding='utf-8'))['data']
    questions = {i: r for (i, r) in enumerate(data)}
    rows = []
    for path in sorted(glob.glob(args.glob)):
        rows.extend(evaluate(path, questions))
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('model\tselector\tdataset\tn\tacc\tinvalid\tavg_ctx\n')
        for r in rows:
            f.write(f"{r['model']}\t{r['selector']}\t{r['dataset']}\t{r['n']}\t{r['acc']:.2f}\t{r['invalid']}\t{r['avg_ctx']:.2f}\n")
    macro = {(r['model'], r['selector']): r['acc'] for r in rows if r['dataset'] == 'MACRO'}
    for r in rows:
        if r['dataset'] == 'ALL':
            print(f"{r['model']:10s} {r['selector']:26s} micro {r['acc']:6.2f}  macro {macro[r['model'], r['selector']]:6.2f}  invalid {r['invalid']:5d}  ctx {r['avg_ctx']:.2f}")
if __name__ == '__main__':
    main()
