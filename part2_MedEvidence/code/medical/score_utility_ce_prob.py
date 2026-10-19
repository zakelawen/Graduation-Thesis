import os
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
RAG_PROBE = os.environ.get('RAG_PROBE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rag_probe'))
sys.path.insert(0, os.path.join(RAG_PROBE, 'Rebuttal/scripts'))
sys.path.insert(0, os.path.join(RAG_PROBE, 'scripts'))
import top10_score_contexts as t
ap = argparse.ArgumentParser()
ap.add_argument('--input', type=Path, required=True)
ap.add_argument('--model', required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--batch-size', type=int, default=128)
ap.add_argument('--dataset', default='mirage')
args = ap.parse_args()
(meta, rows) = t.load_topk(args.input, None)
flat = t.flatten_rows(rows)
tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True).cuda().eval()
scores = []
with torch.inference_mode():
    for s in tqdm(range(0, len(flat), args.batch_size), desc='utility_ce_prob1'):
        b = flat[s:s + args.batch_size]
        enc = tok([x['question'] for x in b], [x['doc_text'] for x in b], padding=True, truncation=True, return_tensors='pt').to('cuda')
        logits = model(**enc).logits.float()
        scores.extend((torch.softmax(logits, -1)[:, 1] if logits.shape[-1] > 1 else logits[:, 0]).cpu().tolist())
details = {'model': args.model, 'backend': 'transformers', 'score': 'softmax(logits)[:, 1] (same as PopQA classification evaluation)'}
v = np.asarray(scores, dtype=float)
payload = {'meta': {'input': str(args.input), 'dataset': args.dataset, 'selector': 'cross_encoder', 'num_questions': len(rows), 'num_docs': len(flat), 'score_mean': float(v.mean()), 'score_min': float(v.min()), 'score_max': float(v.max()), 'details': details}, 'data': t.attach_scores(rows, flat, scores, details)}
t.dump_json(args.output, payload)
print('[saved]', args.output, 'mean', v.mean())
