import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d2rag.judge import judge, make_client
from d2rag.utils import load_jsonl, write_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--answer-field', default='model_output_raw')
    args = ap.parse_args()
    client = make_client()
    for path in args.files:
        rows = load_jsonl(path)
        todo = [r for r in rows if r.get('is_match') in (None, -1)]
        if todo:
            with ThreadPoolExecutor(args.workers) as pool:
                results = pool.map(lambda r: judge(client, r.get('question', ''), r.get('ground_truths', []), r.get(args.answer_field, '')), todo)
                for (r, label) in zip(todo, results):
                    r['is_match'] = label
            write_jsonl(path, rows)
        valid = [r for r in rows if r['is_match'] != -1]
        acc = 100 * sum((r['is_match'] == 1 for r in valid)) / max(len(valid), 1)
        print(f'{os.path.basename(path)}: judged {len(todo)} new, acc {acc:.2f}%  (n={len(rows)}, api_failed={len(rows) - len(valid)})')
if __name__ == '__main__':
    main()
