import os
import json, random, re
import pandas as pd
MT_WORK = os.environ.get('MT_WORK', os.getcwd())
W = MT_WORK
norm = lambda s: re.sub('[^a-z0-9]', '', (s or '').lower())
cur = json.load(open(f'{W}/data/train_questions.json'))
bench = json.load(open(f'{W}/data/mirage/benchmark.json'))
test_q = {norm(v['question']) for items in bench.values() for v in items.values()}
used = {norm(v['question']) for items in cur.values() for v in items.values()}
df = pd.read_parquet(f'{W}/data/train_raw/medmcqa/data/train-00000-of-00001.parquet')
df = df[df['choice_type'] == 'single']
df['n'] = df['question'].map(norm)
df = df[~df['n'].isin(test_q) & ~df['n'].isin(used)].drop_duplicates('n')
add = random.Random(44).sample(df.to_dict('records'), 3000)
extra = {'medmcqa_train': {r['id']: {'question': r['question'], 'options': {'A': r['opa'], 'B': r['opb'], 'C': r['opc'], 'D': r['opd']}, 'answer': 'ABCD'[int(r['cop'])]} for r in add}}
json.dump(extra, open(f'{W}/data/train_questions_extra.json', 'w'), ensure_ascii=False)
cur['medmcqa_train'].update(extra['medmcqa_train'])
json.dump(cur, open(f'{W}/data/train_questions.json', 'w'), ensure_ascii=False)
st = json.load(open(f'{W}/data/train_questions.stats.json'))
st['medmcqa_train']['extended_by_2'] = 3000
st['medmcqa_train']['sampled'] = len(cur['medmcqa_train'])
json.dump(st, open(f'{W}/data/train_questions.stats.json', 'w'), indent=1)
json.dump([f'medmcqa_train:{k}' for k in cur['medmcqa_train']], open(f'{W}/labels/train_medmcqa_ids.json', 'w'))
print({k: len(v) for (k, v) in cur.items()})
