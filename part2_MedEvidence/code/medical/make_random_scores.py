import os
import json
import numpy as np
MT_WORK = os.environ.get('MT_WORK', os.getcwd())
W = MT_WORK
obj = json.load(open(f'{W}/scores/retriever_score.json', encoding='utf-8'))
rng = np.random.default_rng(0)
for row in obj['data']:
    for doc in row['scored_docs']:
        doc['score'] = float(rng.random())
        doc['detail'] = {}
obj['meta'] = {**{k: v for (k, v) in obj['meta'].items() if k in ('input', 'dataset', 'num_questions', 'num_docs')}, 'selector': 'random', 'seed': 0}
json.dump(obj, open(f'{W}/scores/random_seed0.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('saved', f'{W}/scores/random_seed0.json', len(obj['data']))
