import os
import argparse
import importlib.util
import json
import sys
import types
from collections import Counter
from pathlib import Path
RAG_PROBE = os.environ.get('RAG_PROBE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rag_probe'))
MT_WORK = os.environ.get('MT_WORK', os.getcwd())
W = Path(MT_WORK)
STEP4B = Path(os.path.join(RAG_PROBE, 'scripts/step4b_build_retrieved_dataset.py'))

def load_step4b():
    cfg = types.ModuleType('config')
    for name in ('RETRIEVED_DIR', 'ANNOTATED_DIR', 'FINAL_DIR', 'DATA_DIR', 'PROCESSED_DIR'):
        setattr(cfg, name, '')
    cfg.DATASETS = ()
    cfg.TRAIN_RATIO = 0.8
    sys.modules['config'] = cfg
    spec = importlib.util.spec_from_file_location('step4b', STEP4B)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrieval', default=str(W / 'data/train_medcpt_top10.json'))
    ap.add_argument('--labels', default=str(W / 'labels/train_top1_claude.jsonl'))
    ap.add_argument('--out-dir', default=str(W / 'data/medical_retrieved'))
    ap.add_argument('--train-ratio', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    step4b = load_step4b()
    rows = {f"{r['dataset']}:{r['id']}": r for r in json.load(open(args.retrieval, encoding='utf-8'))['data']}
    labels = {}
    for line in open(args.labels, encoding='utf-8'):
        rec = json.loads(line)
        if rec.get('status') == 'ok':
            labels[rec['id']] = rec
    (samples, dropped) = ([], Counter())
    for (key, lab) in labels.items():
        r = rows.get(key)
        if r is None or r['retrieved_docs'][0]['id'] != lab['doc_id']:
            dropped['stale'] += 1
            continue
        if lab['label_name'] not in ('relevant', 'distracting'):
            dropped[lab['label_name']] += 1
            continue
        d = r['retrieved_docs'][0]
        samples.append({'question': r['question'], 'answers': r['answers'], 'doc': {'title': d['title'], 'text': d['text'], 'score': d['score'], 'passage_id': d['id'], 'corpus': d['corpus']}, 'label': int(lab['label_name'] == 'relevant'), 'label_name': lab['label_name'], 'annotation': {'label': lab['label_name'], 'confidence': lab['confidence'], 'model': lab.get('model')}, 'source_dataset': r['dataset'], 'original_index': r['id']})
    (train_unbal, eval_) = step4b.stratified_split(samples, args.train_ratio, args.seed)
    (train, details) = step4b.balance_train_by_dataset(train_unbal, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for (name, obj) in (('train_unbalanced', train_unbal), ('train', train), ('eval', eval_)):
        json.dump(obj, open(out / f'{name}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    count = lambda xs: dict(Counter((f"{x['source_dataset']}|{x['label_name']}" for x in xs)))
    stats = {'description': 'Medical retrieved-top1 relevant vs distracting dataset (MedCPT top-1, Claude Opus 5.5 labels)', 'source': {'retrieval': args.retrieval, 'labels': args.labels}, 'rules': 'step4b: drop neutral; stratified split by (source, label); balance train by downsampling majority per source; eval natural', 'seed': args.seed, 'train_ratio': args.train_ratio, 'dropped': dict(dropped), 'counts': {'all': count(samples), 'train_unbalanced': count(train_unbal), 'train': count(train), 'eval': count(eval_)}, 'balance_details': details}
    json.dump(stats, open(out / 'stats.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    print(f'samples {len(samples)} | train {len(train)} | eval {len(eval_)} | dropped {dict(dropped)}')
    print('train:', count(train))
    print('eval:', count(eval_))
if __name__ == '__main__':
    main()
