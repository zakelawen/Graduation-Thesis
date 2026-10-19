import argparse
import json
import os
import random
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
RAG_PROBE = os.environ.get('RAG_PROBE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rag_probe'))
MT_WORK = os.environ.get('MT_WORK', os.getcwd())
W = MT_WORK
ENV = os.path.join(MT_WORK, '.env')
FULL_SCHEMA = '{\n  "label": "relevant" | "distracting" | "neutral",\n  "confidence": 0.00-1.00,\n  "rationale": "<one- or two-sentence justification>",\n  "supporting_spans": ["<short excerpt(s) from the document that justify the judgment>"],\n  "inference_type": "direct" | "indirect" | "multi-hop" | "contradiction"\n}'
SHORT_SCHEMA = '{\n  "label": "relevant" | "distracting" | "neutral",\n  "confidence": 0.00-1.00\n}'
FULL_RULE = 'Provide only the required structured output (JSON, see schema below) and a concise 1-2 sentence rationale (no internal reasoning steps).'
SHORT_RULE = 'Provide only the required structured output (JSON, see schema below); do not add any rationale or other text.'
VALID = ('relevant', 'distracting', 'neutral')

def short_prompt_template():
    cfg = types.ModuleType('config')
    cfg.RETRIEVED_DIR = cfg.ANNOTATED_DIR = cfg.GPT_API_URL = cfg.GPT_API_KEY = cfg.GPT_MODEL = ''
    cfg.DATASETS = ()
    sys.modules['config'] = cfg
    sys.path.insert(0, os.path.join(RAG_PROBE, 'scripts'))
    import step3_annotate
    p = step3_annotate.CLASSIFICATION_PROMPT
    assert p.count(FULL_SCHEMA) == 1 and p.count(FULL_RULE) == 1
    p = p.replace(FULL_SCHEMA, SHORT_SCHEMA).replace(FULL_RULE, SHORT_RULE)
    p = p.replace('• Provide supporting_spans identifying the explicit sentence(s) or fact(s).\n', '')
    return p

def build(template, question, answers, document):
    return template.replace('{question}', question).replace('{answers}', json.dumps(answers, ensure_ascii=False)).replace('{document}', document)

def parse(content):
    c = content.strip()
    if c.startswith('```'):
        c = c.split('\n', 1)[1].rsplit('```', 1)[0]
    (i, j) = (c.find('{'), c.rfind('}'))
    obj = json.loads(c[i:j + 1])
    label = str(obj.get('label', '')).strip().lower()
    if label not in VALID:
        raise ValueError(f'bad label {label!r}')
    return (label, float(obj.get('confidence', 0) or 0))

def rows_from(retrieval_path):
    data = json.load(open(retrieval_path, encoding='utf-8'))['data']
    out = []
    for (idx, r) in enumerate(data):
        d = r['retrieved_docs'][0]
        out.append({'idx': idx, 'id': f"{r['dataset']}:{r['id']}", 'dataset': r['dataset'], 'question': r['question'], 'answers': [f"{r['answer']}. {r['options'][r['answer']]}"], 'document': f"{d['title']}\n{d['text']}", 'doc_id': d['id'], 'corpus': d['corpus']})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrieval', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--ids', default=None)
    ap.add_argument('--model', default='claude-opus-5-5')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=80)
    ap.add_argument('--retries', type=int, default=4)
    args = ap.parse_args()
    env = dict((l.strip().split('=', 1) for l in open(ENV) if '=' in l and (not l.startswith('#'))))
    (key, base) = (env['KAKOU_API_KEY'], env['KAKOU_BASE_URL'].rstrip('/'))
    template = short_prompt_template()
    os.makedirs(f'{W}/prompts', exist_ok=True)
    open(f'{W}/prompts/annotation_short.txt', 'w', encoding='utf-8').write(template)
    rows = rows_from(args.retrieval)
    if args.ids:
        wanted = set(json.load(open(args.ids, encoding='utf-8')))
        rows = [r for r in rows if r['id'] in wanted]
    done = {}
    if os.path.exists(args.output):
        for line in open(args.output, encoding='utf-8'):
            rec = json.loads(line)
            if rec.get('status') == 'ok':
                done[rec['id']] = rec['doc_id']
    todo = [r for r in rows if done.get(r['id']) != r['doc_id']]
    print(f'rows {len(rows)}, already labeled {len(done)}, to label {len(todo)}', flush=True)
    session = requests.Session()
    session.trust_env = False
    lock = threading.Lock()
    stop = threading.Event()
    usage = {'in': 0, 'out': 0, 'ok': 0, 'err': 0}

    def call(r):
        if stop.is_set():
            return None
        payload = {'model': args.model, 'messages': [{'role': 'user', 'content': build(template, r['question'], r['answers'], r['document'])}], 'temperature': 0, 'max_tokens': args.max_tokens}
        err = None
        for attempt in range(args.retries + 1):
            try:
                resp = session.post(base + '/chat/completions', headers={'Authorization': f'Bearer {key}'}, json=payload, timeout=180)
                if resp.status_code in (401, 402, 403):
                    stop.set()
                    return {'id': r['id'], 'status': 'fatal', 'error': resp.text[:300]}
                if resp.status_code == 429 or resp.status_code >= 500:
                    err = f'HTTP {resp.status_code}'
                    time.sleep(2 ** attempt + random.random())
                    continue
                j = resp.json()
                (label, conf) = parse(j['choices'][0]['message']['content'])
                u = j.get('usage', {})
                return {'idx': r['idx'], 'id': r['id'], 'dataset': r['dataset'], 'doc_id': r['doc_id'], 'corpus': r['corpus'], 'status': 'ok', 'label': int(label == 'relevant'), 'label_name': label, 'confidence': conf, 'prompt_tokens': u.get('prompt_tokens', 0), 'completion_tokens': u.get('completion_tokens', 0), 'model': j.get('model')}
            except Exception as e:
                err = repr(e)[:200]
                time.sleep(2 ** attempt)
        return {'id': r['id'], 'status': 'error', 'error': err}
    t0 = time.time()
    with open(args.output, 'a', encoding='utf-8') as out, ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(call, r) for r in todo]
        for (n, f) in enumerate(as_completed(futures), 1):
            rec = f.result()
            if rec is None:
                continue
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out.flush()
                if rec['status'] == 'ok':
                    usage['ok'] += 1
                    usage['in'] += rec['prompt_tokens']
                    usage['out'] += rec['completion_tokens']
                else:
                    usage['err'] += 1
            if n % 50 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} ok={usage['ok']} err={usage['err']} tokens in={usage['in']} out={usage['out']} ({time.time() - t0:.0f}s)", flush=True)
    if stop.is_set():
        print('STOPPED: auth/quota error', flush=True)
    print('LABEL DONE', json.dumps(usage), flush=True)
if __name__ == '__main__':
    main()
