import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from config import ANNOTATED_DIR, DATASETS, TRAIN_RATIO
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    ANNOTATED_DIR = str(PROJECT_ROOT / 'data' / 'annotated')
    DATASETS = ['nq', 'triviaqa']
    TRAIN_RATIO = 0.8
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'final_retrieved'
LABEL_TO_ID = {'distracting': 0, 'relevant': 1}
VALID_LABELS = set(LABEL_TO_ID.keys())

def normalize_label(label: Any) -> str:
    if label is None:
        return 'unknown'
    return str(label).strip().lower()

def has_valid_doc(doc: Any) -> bool:
    if isinstance(doc, dict):
        return bool(str(doc.get('text', '')).strip())
    return bool(str(doc).strip())

def load_annotated_file(path: Path) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f'Expected a list in {path}, got {type(data)}')
    return data

def build_samples_for_dataset(dataset_name: str, annotated_dir: Path) -> tuple[list[dict], dict]:
    path = annotated_dir / f'{dataset_name}_annotated.json'
    if not path.exists():
        print(f'[skip] Missing {path}')
        return ([], {'path': str(path), 'exists': False, 'total_raw': 0, 'annotation_label_counts': {}, 'kept_label_counts': {}, 'dropped_counts': {}})
    raw = load_annotated_file(path)
    annotation_label_counts = Counter()
    kept_label_counts = Counter()
    dropped_counts = Counter()
    samples: list[dict] = []
    for (idx, item) in enumerate(raw):
        ann = item.get('annotation', {}) or {}
        label_name = normalize_label(ann.get('label', 'unknown'))
        annotation_label_counts[label_name] += 1
        if label_name not in VALID_LABELS:
            dropped_counts[f'label={label_name}'] += 1
            continue
        doc = item.get('retrieved_top1')
        if not has_valid_doc(doc):
            dropped_counts['missing_or_empty_retrieved_top1'] += 1
            continue
        if 'question' not in item:
            dropped_counts['missing_question'] += 1
            continue
        sample = {'question': item['question'], 'answers': item.get('answers', []), 'doc': doc, 'label': LABEL_TO_ID[label_name], 'label_name': label_name, 'annotation': ann, 'source_dataset': dataset_name, 'original_index': idx}
        samples.append(sample)
        kept_label_counts[label_name] += 1
    stats = {'path': str(path), 'exists': True, 'total_raw': len(raw), 'annotation_label_counts': dict(sorted(annotation_label_counts.items())), 'kept_label_counts': dict(sorted(kept_label_counts.items())), 'dropped_counts': dict(sorted(dropped_counts.items())), 'kept_total': len(samples)}
    return (samples, stats)

def stratified_split(samples: list[dict], train_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in samples:
        groups[s.get('source_dataset', 'unknown'), s.get('label_name', 'unknown')].append(s)
    (train, eval_) = ([], [])
    for key in sorted(groups.keys()):
        group = list(groups[key])
        rng.shuffle(group)
        if len(group) == 1:
            n_train = 1
        else:
            n_train = int(len(group) * train_ratio)
            n_train = max(1, min(n_train, len(group) - 1))
        train.extend(group[:n_train])
        eval_.extend(group[n_train:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return (train, eval_)

def balance_train_by_dataset(train: list[dict], seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    by_dataset_label: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in train:
        ds = str(s.get('source_dataset', 'unknown'))
        label_name = str(s.get('label_name', 'unknown'))
        by_dataset_label[ds, label_name].append(s)
    datasets = sorted({ds for (ds, _) in by_dataset_label.keys()})
    balanced: list[dict] = []
    details: dict[str, dict] = {}
    for ds in datasets:
        rel = list(by_dataset_label.get((ds, 'relevant'), []))
        dis = list(by_dataset_label.get((ds, 'distracting'), []))
        n_keep = min(len(rel), len(dis))
        if n_keep == 0:
            details[ds] = {'before': {'relevant': len(rel), 'distracting': len(dis)}, 'after': {'relevant': 0, 'distracting': 0}, 'dropped': {'relevant': len(rel), 'distracting': len(dis)}, 'warning': 'one class is missing; dataset contributes 0 balanced samples'}
            continue
        rng.shuffle(rel)
        rng.shuffle(dis)
        kept_rel = rel[:n_keep]
        kept_dis = dis[:n_keep]
        balanced.extend(kept_rel)
        balanced.extend(kept_dis)
        details[ds] = {'before': {'relevant': len(rel), 'distracting': len(dis)}, 'after': {'relevant': len(kept_rel), 'distracting': len(kept_dis)}, 'dropped': {'relevant': len(rel) - len(kept_rel), 'distracting': len(dis) - len(kept_dis)}}
    rng.shuffle(balanced)
    return (balanced, details)

def count_by(items: list[dict], keys: list[str]) -> dict:
    if not keys:
        return {'count': len(items)}
    key = keys[0]
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        buckets[str(item.get(key, 'unknown'))].append(item)
    return {k: count_by(v, keys[1:]) for (k, v) in sorted(buckets.items())}

def label_counts(items: list[dict]) -> dict:
    c = Counter((s.get('label_name', 'unknown') for s in items))
    out = dict(sorted(c.items()))
    out['total'] = len(items)
    return out

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def print_distribution(title: str, items: list[dict]):
    print(f'\n===== {title} =====')
    print(f'Total: {len(items)}')
    by_dataset = defaultdict(list)
    for s in items:
        by_dataset[s.get('source_dataset', 'unknown')].append(s)
    for ds in sorted(by_dataset.keys()):
        subset = by_dataset[ds]
        c = Counter((x.get('label_name', 'unknown') for x in subset))
        rel = c.get('relevant', 0)
        dis = c.get('distracting', 0)
        total = len(subset)
        rel_rate = rel / total if total else 0.0
        print(f'  {ds:8s}: total={total:5d}  relevant={rel:5d}  distracting={dis:5d}  relevant_rate={rel_rate:.3f}')
    c_all = Counter((x.get('label_name', 'unknown') for x in items))
    rel = c_all.get('relevant', 0)
    dis = c_all.get('distracting', 0)
    total = len(items)
    rel_rate = rel / total if total else 0.0
    print(f"  {'combined':8s}: total={total:5d}  relevant={rel:5d}  distracting={dis:5d}  relevant_rate={rel_rate:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Step4b: build retrieved-top1 relevant/distracting dataset. Default: balanced TRAIN, natural EVAL.')
    parser.add_argument('--annotated_dir', type=str, default=ANNOTATED_DIR, help='Directory containing {dataset}_annotated.json')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR), help='Output directory, default data/final_retrieved')
    parser.add_argument('--train_ratio', type=float, default=float(TRAIN_RATIO), help='Train split ratio, default from config.TRAIN_RATIO')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--datasets', type=str, default=','.join(DATASETS), help='Comma-separated datasets, default from config.DATASETS')
    parser.add_argument('--no_balance_train', action='store_true', help='Do not downsample training split. By default train.json is balanced per dataset.')
    args = parser.parse_args()
    annotated_dir = Path(args.annotated_dir)
    output_dir = Path(args.output_dir)
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    balance_train = not args.no_balance_train
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError(f'--train_ratio must be in (0,1), got {args.train_ratio}')
    print('=== Step4b: Build retrieved-doc dataset ===')
    print(f'annotated_dir = {annotated_dir}')
    print(f'output_dir    = {output_dir}')
    print(f'datasets      = {datasets}')
    print(f'train_ratio   = {args.train_ratio}')
    print(f'seed          = {args.seed}')
    print("filter        = annotation.label in {'relevant','distracting'}; NO confidence filter")
    print(f'balance_train = {balance_train} (train only; eval remains natural)')
    all_samples: list[dict] = []
    per_dataset_stats = {}
    for ds in datasets:
        (samples, stats) = build_samples_for_dataset(ds, annotated_dir)
        all_samples.extend(samples)
        per_dataset_stats[ds] = stats
        print(f"\n[{ds}] raw={stats['total_raw']} kept={stats.get('kept_total', 0)}")
        print(f"  annotation labels: {stats['annotation_label_counts']}")
        print(f"  kept labels:       {stats['kept_label_counts']}")
        print(f"  dropped:           {stats['dropped_counts']}")
    if not all_samples:
        raise RuntimeError('No samples were kept. Check annotated files and label names.')
    (train_unbalanced, eval_) = stratified_split(all_samples, args.train_ratio, args.seed)
    if balance_train:
        (train, balance_details) = balance_train_by_dataset(train_unbalanced, args.seed)
    else:
        train = train_unbalanced
        balance_details = {}
    stats = {'description': 'Retrieved-top1 relevant vs retrieved-top1 distracting dataset for Step11 gate.', 'filter': {'kept_labels': sorted(VALID_LABELS), 'confidence_filter': None, 'balanced_train': balance_train, 'eval_is_natural_unbalanced': True, 'balancing_rule': 'within train split, downsample majority class separately per source_dataset' if balance_train else None}, 'seed': args.seed, 'train_ratio': args.train_ratio, 'per_dataset_raw': per_dataset_stats, 'balance_details': balance_details, 'all_distribution': count_by(all_samples, ['source_dataset', 'label_name']), 'train_unbalanced_distribution': count_by(train_unbalanced, ['source_dataset', 'label_name']), 'train_distribution': count_by(train, ['source_dataset', 'label_name']), 'eval_distribution': count_by(eval_, ['source_dataset', 'label_name']), 'summary': {'all': label_counts(all_samples), 'train_unbalanced': label_counts(train_unbalanced), 'train': label_counts(train), 'eval': label_counts(eval_)}, 'schema': {'question': 'str', 'answers': 'list | str, kept from annotated file', 'doc': 'retrieved_top1 dict', 'label': '1=relevant, 0=distracting', 'label_name': 'relevant | distracting', 'annotation': 'original Step3 annotation dict', 'source_dataset': 'nq | triviaqa', 'original_index': 'index in original annotated json'}}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'all.json', all_samples)
    write_json(output_dir / 'train_unbalanced.json', train_unbalanced)
    write_json(output_dir / 'train.json', train)
    write_json(output_dir / 'eval.json', eval_)
    write_json(output_dir / 'stats.json', stats)
    print_distribution('ALL / natural', all_samples)
    print_distribution('TRAIN_UNBALANCED / natural split', train_unbalanced)
    print_distribution('TRAIN / used for training', train)
    print_distribution('EVAL / natural', eval_)
    if balance_train:
        print('\n===== TRAIN BALANCING DETAILS =====')
        for (ds, detail) in sorted(balance_details.items()):
            print(f"  {ds}: before={detail['before']} after={detail['after']} dropped={detail['dropped']}")
    print('\nSaved:')
    print(f"  {output_dir / 'all.json'}")
    print(f"  {output_dir / 'train_unbalanced.json'}")
    print(f"  {output_dir / 'train.json'}")
    print(f"  {output_dir / 'eval.json'}")
    print(f"  {output_dir / 'stats.json'}")
    print('\nNext:')
    print('  1) Inspect data/final_retrieved/stats.json')
    print('  2) Train retrieved-doc CSV scorer on data/final_retrieved/train.json')
    print('  3) Evaluate on data/final_retrieved/eval.json')
if __name__ == '__main__':
    main()
