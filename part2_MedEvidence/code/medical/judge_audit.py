import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import requests
MT_WORK = os.environ.get('MT_WORK', os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_mcq import predict_letter
from extract_invalid_llm import load_env
W = MT_WORK
PROMPT = 'You are grading a model\'s answer to a multiple-choice question.\n\nQuestion:\n{question}\n\nCorrect answer: {gold}. {gold_text}\n\nModel response:\n{response}\n\nIs the model response correct? It is correct only if it clearly selects the correct option (by letter or by its content) and does not select any other option.\nReply with JSON only: {{"correct": true or false}}'

def judge(session, env, model, q, response):
    body = {'model': model, 'temperature': 0, 'max_tokens': 20, 'messages': [{'role': 'user', 'content': PROMPT.format(question=q['question'], gold=q['answer'], gold_text=q['options'][q['answer']], response=response)}]}
    for attempt in range(5):
        try:
            r = session.post(env['XTY_BASE_URL'] + '/v1/chat/completions', json=body, timeout=60, headers={'Authorization': 'Bearer ' + env['XTY_API_KEY']})
            if r.status_code in (401, 402, 403):
                raise SystemExit(f'auth/billing error {r.status_code}: {r.text[:200]}')
            r.raise_for_status()
            text = r.json()['choices'][0]['message']['content']
            m = re.search('"correct"\\s*:\\s*(true|false)', text, re.I)
            return (m.group(1).lower() == 'true' if m else None, text)
        except SystemExit:
            raise
        except Exception as e:
            err = str(e)
    return (None, err[:200])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt-4.1-mini')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--out', default=f'{W}/results/llm_judge_audit.jsonl')
    args = ap.parse_args()
    env = load_env()
    data = json.load(open(f'{W}/data/mirage_medcpt_top10.json', encoding='utf-8'))['data']
    pool = []
    for path in sorted(glob.glob(f'{W}/generations/*_mirage_*.json')):
        obj = json.load(open(path, encoding='utf-8'))
        for out in obj['outputs']:
            idx = int(out['idx'])
            pred = out['generation']['prediction']
            letter = predict_letter(pred, data[idx]['options'])
            if letter is not None:
                pool.append((obj['meta']['model'], obj['meta']['selector'], idx, pred, letter))
    random.seed(0)
    sample = random.sample(pool, args.n)
    print(f'rule-parsed outputs: {len(pool)}; auditing {len(sample)}')
    session = requests.Session()
    session.trust_env = False

    def work(item):
        (model, selector, idx, pred, letter) = item
        q = data[idx]
        (verdict, raw) = judge(session, env, args.model, q, pred)
        return {'model': model, 'selector': selector, 'idx': idx, 'dataset': q['dataset'], 'prediction': pred, 'rule_letter': letter, 'gold': q['answer'], 'rule_correct': letter == q['answer'], 'llm_correct': verdict, 'raw': raw}
    with ThreadPoolExecutor(8) as ex:
        recs = list(ex.map(work, sample))
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    ok = [r for r in recs if r['llm_correct'] is not None]
    agree = sum((r['rule_correct'] == r['llm_correct'] for r in ok))
    print(f'judged {len(ok)}/{len(recs)}; agreement {agree}/{len(ok)} = {100 * agree / len(ok):.2f}%')
    print('rule acc on sample %.2f, LLM acc on sample %.2f' % (100 * sum((r['rule_correct'] for r in ok)) / len(ok), 100 * sum((r['llm_correct'] for r in ok)) / len(ok)))
    print('output forms:', Counter(('letter-only' if re.fullmatch('\\s*\\(?[A-E]\\)?[.:]?\\s*', r['prediction']) else 'other' for r in recs)))
    for r in ok:
        if r['rule_correct'] != r['llm_correct']:
            print(f"  DISAGREE [{r['model']}/{r['selector']}/{r['dataset']}] pred={r['prediction'][:80]!r} rule={r['rule_letter']} gold={r['gold']} rule_correct={r['rule_correct']} llm_correct={r['llm_correct']}")
if __name__ == '__main__':
    main()
