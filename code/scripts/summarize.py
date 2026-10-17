import argparse
import collections
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d2rag.utils import load_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    args = ap.parse_args()
    for path in sorted(args.files):
        rows = load_jsonl(path)
        by = collections.defaultdict(lambda : [0, 0])
        for r in rows:
            s = r.get('decoding_strategy') or r.get('strategy')
            by[s][0] += 1
            by[s][1] += str(r.get('is_match')) == '1'
        judged = [r for r in rows if r.get('is_match') not in (None, -1)]
        acc = 100 * sum((str(r['is_match']) == '1' for r in judged)) / max(len(judged), 1)
        detail = '  '.join((f'{k}:{n}/{100 * c / n:.1f}%' for (k, (n, c)) in sorted(by.items())))
        print(f'{acc:6.2f}  n={len(rows)} judged={len(judged)}  {detail}  {os.path.basename(path)}')
if __name__ == '__main__':
    main()
