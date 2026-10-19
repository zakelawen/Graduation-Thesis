import os
import json, random, re
import pandas as pd
MT_WORK = os.environ.get('MT_WORK', os.getcwd())
W = MT_WORK
norm = lambda s: re.sub('[^a-z0-9]', '', (s or '').lower())
bench = json.load(open(f'{W}/data/mirage/benchmark.json'))
test_q = {norm(v['question']) for items in bench.values() for v in items.values()}
rng = random.Random(42)
(out, stats) = ({}, {})
mq = [json.loads(l) for l in open(f'{W}/data/train_raw/MedQA-USMLE-4-options/phrases_no_exclude_train.jsonl')]
mq_ok = [r for r in mq if norm(r['question']) not in test_q]
pick = rng.sample(mq_ok, 3000)
out['medqa_train'] = {f'mq{i:05d}': {'question': r['question'], 'options': r['options'], 'answer': r['answer_idx']} for (i, r) in enumerate(pick)}
stats['medqa_train'] = dict(pool=len(mq), after_overlap_filter=len(mq_ok), sampled=len(pick))
df = pd.read_parquet(f'{W}/data/train_raw/medmcqa/data/train-00000-of-00001.parquet')
df = df[df['choice_type'] == 'single']
df['n'] = df['question'].map(norm)
df = df[~df['n'].isin(test_q)].drop_duplicates('n')
rows = df.to_dict('records')
pick = rng.sample(rows, 3000)
out['medmcqa_train'] = {r['id']: {'question': r['question'], 'options': {'A': r['opa'], 'B': r['opb'], 'C': r['opc'], 'D': r['opd']}, 'answer': 'ABCD'[int(r['cop'])]} for r in pick}
stats['medmcqa_train'] = dict(single_answer_dedup_no_overlap=len(rows), sampled=len(pick))
json.dump(out, open(f'{W}/data/train_questions.json', 'w'), ensure_ascii=False)
json.dump(stats, open(f'{W}/data/train_questions.stats.json', 'w'), indent=1)
print(stats)
print(next(iter(out['medqa_train'].values())))
print(next(iter(out['medmcqa_train'].values())))
