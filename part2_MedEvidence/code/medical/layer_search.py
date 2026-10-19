import argparse
import json
from pathlib import Path

def load_sweeps(run_name, dirs):
    merged = {}
    for d in dirs:
        path = Path(d) / f'{run_name}_csv_sweep.json'
        if path.exists():
            for (key, entry) in json.loads(path.read_text(encoding='utf-8')).items():
                merged[key] = entry
        for path in sorted(Path(d).glob(f'{run_name}_inject_*_csv_result.json')):
            entry = json.loads(path.read_text(encoding='utf-8'))
            if Path(entry['checkpoint']).exists():
                merged[f"inject_{entry['str_layer']}_cls_{entry['cls_layer']}"] = entry
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--inject', default=None)
    ap.add_argument('--cls', default=None)
    ap.add_argument('--merge-to', default=None)
    args = ap.parse_args()
    sweep = load_sweeps(args.run_name, args.dirs)
    if not sweep:
        raise SystemExit(f'no sweep results for {args.run_name} in {args.dirs}')
    best_key = max(sweep, key=lambda k: sweep[k]['best_auroc'])
    best = sweep[best_key]
    ranked = sorted(sweep.items(), key=lambda kv: -kv[1]['best_auroc'])
    for (key, e) in ranked[:5]:
        print(f"{key:24s} val AUROC {100 * e['best_auroc']:.2f}  acc {100 * e['best_accuracy']:.2f}  epoch {e['best_epoch']}")
    if args.inject and args.cls:
        inj = [int(x) for x in args.inject.split(',')]
        cls = [int(x) for x in args.cls.split(',')]
        edge = best['str_layer'] in (min(inj), max(inj)) or best['cls_layer'] in (min(cls), max(cls))
        print('BOUNDARY' if edge else 'INTERIOR', best_key)
    if args.merge_to:
        out = Path(args.merge_to)
        out.mkdir(parents=True, exist_ok=True)
        (out / f'{args.run_name}_csv_sweep.json').write_text(json.dumps(sweep, indent=2) + '\n', encoding='utf-8')
        print(f'merged {len(sweep)} configs -> {out}; best {best_key}')
if __name__ == '__main__':
    main()
