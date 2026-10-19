import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import requests
MT_WORK = os.environ.get('MT_WORK', os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_mcq import predict_letter
W = MT_WORK
PROMPT = 'Below are the answer options of a multiple-choice question and a model\'s response. Decide which single option the response selects.\nRules:\n- Choose an option only if the response itself explicitly states it, by letter or by stating that option\'s content.\n- Answer NONE if the response selects no option, selects several options, is off-topic, restates a question or passage, or is empty.\n- Do not try to answer the question yourself, and do not judge correctness.\n- "evidence" must be copied exactly from the response and must show the selected option; use "" for NONE.\n\nOptions:\n{options}\n\nResponse:\n{response}\n\nReply with JSON only: {{"choice": "<letter or NONE>", "evidence": "<exact span from the response>"}}'

def norm_text(t):
    return re.sub('\\s+', ' ', t).strip().lower()

def load_env():
    env = {}
    for line in open(f'{W}/.env', encoding='utf-8'):
        if '=' in line:
            (k, v) = line.strip().split('=', 1)
            env[k] = v
    return env

def ask(session, env, model, options, response, letters):
    if not response.strip():
        return ('NONE', 'empty response')
    opts = '\n'.join((f'{k}. {v}' for (k, v) in options.items()))
    body = {'model': model, 'temperature': 0, 'max_tokens': 120, 'messages': [{'role': 'user', 'content': PROMPT.format(options=opts, response=response)}]}
    for attempt in range(5):
        try:
            r = session.post(env['XTY_BASE_URL'] + '/v1/chat/completions', json=body, timeout=60, headers={'Authorization': 'Bearer ' + env['XTY_API_KEY']})
            if r.status_code in (401, 402, 403):
                raise SystemExit(f'auth/billing error {r.status_code}: {r.text[:200]}')
            r.raise_for_status()
            text = r.json()['choices'][0]['message']['content']
            try:
                obj = json.loads(re.search('\\{.*\\}', text, re.S).group(0))
                (choice, evidence) = (str(obj.get('choice', '')).strip().upper(), str(obj.get('evidence', '')))
            except Exception:
                return ('PARSE_ERROR', text)
            if choice == 'NONE':
                return ('NONE', text)
            if choice not in letters:
                return ('PARSE_ERROR', text)
            if not evidence.strip() or norm_text(evidence) not in norm_text(response):
                return ('NONE', 'evidence not in response: ' + text)
            return (choice, text)
        except SystemExit:
            raise
        except Exception as e:
            err = e
            time.sleep(2 * (attempt + 1))
    return ('ERROR', str(err)[:200])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt-4.1-mini')
    ap.add_argument('--glob', default=f'{W}/generations/*_mirage_*.json')
    ap.add_argument('--cache', default=f'{W}/results/llm_extract_cache.jsonl')
    ap.add_argument('--out', default=f'{W}/results/downstream_llm_extract.tsv')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    env = load_env()
    data = json.load(open(f'{W}/data/mirage_medcpt_top10.json', encoding='utf-8'))['data']
    cache = {}
    if os.path.exists(args.cache):
        for line in open(args.cache, encoding='utf-8'):
            rec = json.loads(line)
            if rec['choice'] not in ('ERROR',):
                cache[rec['key']] = rec
    files = sorted(glob.glob(args.glob))
    pending = {}
    for path in files:
        for out in json.load(open(path, encoding='utf-8'))['outputs']:
            idx = int(out['idx'])
            pred = out['generation']['prediction']
            if predict_letter(pred, data[idx]['options']) is None:
                key = hashlib.sha1(f'v2|{args.model}|{idx}|{pred}'.encode()).hexdigest()
                if key not in cache:
                    pending[key] = (idx, pred)
    todo = list(pending.items())[:args.limit] if args.limit else list(pending.items())
    print(f'invalid outputs needing LLM: {len(pending)} unique (running {len(todo)}), cached {len(cache)}')
    session = requests.Session()
    session.trust_env = False

    def work(item):
        (key, (idx, pred)) = item
        q = data[idx]
        (choice, raw) = ask(session, env, args.model, q['options'], pred, set(q['options']))
        return {'key': key, 'model': args.model, 'idx': idx, 'prediction': pred, 'choice': choice, 'raw': raw}
    with open(args.cache, 'a', encoding='utf-8') as f, ThreadPoolExecutor(args.workers) as pool:
        for rec in pool.map(work, todo):
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            if rec['choice'] not in ('ERROR',):
                cache[rec['key']] = rec
            if args.limit:
                print(f"  [{data[rec['idx']]['dataset']}] {rec['prediction'][:70]!r} -> {rec['choice']} (gold {data[rec['idx']]['answer']})")
    if args.limit:
        return
    rows = []
    for path in files:
        obj = json.load(open(path, encoding='utf-8'))
        meta = obj['meta']
        by_ds = defaultdict(lambda : [0, 0, 0, 0, 0])
        for out in obj['outputs']:
            idx = int(out['idx'])
            q = data[idx]
            pred = out['generation']['prediction']
            rule = predict_letter(pred, q['options'])
            s = by_ds[q['dataset']]
            s[0] += 1
            s[1] += rule == q['answer']
            if rule is None:
                rec = cache.get(hashlib.sha1(f'v2|{args.model}|{idx}|{pred}'.encode()).hexdigest())
                llm = rec['choice'] if rec else 'MISSING'
                s[3] += 1
                s[4] += llm in q['options']
                s[2] += llm == q['answer']
            else:
                s[2] += rule == q['answer']
        tot = [sum((v[i] for v in by_ds.values())) for i in range(5)]
        for (ds, v) in sorted(by_ds.items()) + [('ALL', tot)]:
            rows.append([meta['model'], meta['selector'], ds, v[0], 100 * v[1] / v[0], 100 * v[2] / v[0], v[3], v[4]])
        macro_rule = sum((100 * v[1] / v[0] for v in by_ds.values())) / len(by_ds)
        macro_llm = sum((100 * v[2] / v[0] for v in by_ds.values())) / len(by_ds)
        rows.append([meta['model'], meta['selector'], 'MACRO', tot[0], macro_rule, macro_llm, tot[3], tot[4]])
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['model', 'selector', 'dataset', 'n', 'acc_rule', 'acc_rule_plus_llm', 'rule_invalid', 'llm_mapped'])
        for r in rows:
            w.writerow([f'{x:.2f}' if isinstance(x, float) else x for x in r])
    for r in rows:
        if r[2] in ('ALL', 'MACRO'):
            print(f'{r[0]:9s} {r[1]:24s} {r[2]:5s} rule {r[4]:6.2f}  rule+LLM {r[5]:6.2f}  invalid {r[6]:4d}  mapped {r[7]:4d}')
if __name__ == '__main__':
    main()
