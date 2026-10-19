import argparse
import glob
import json
import os
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
QUERY = 'ncbi/MedCPT-Query-Encoder'
CORPORA = ('textbooks', 'statpearls', 'pubmed')

def mirage_rows(path):
    bench = json.load(open(path, encoding='utf-8'))
    rows = []
    for (dataset, items) in bench.items():
        for (qid, item) in items.items():
            opts = item['options']
            question_with_options = item['question'].strip() + '\n' + '\n'.join((f'{k}. {v}' for (k, v) in opts.items()))
            rows.append({'dataset': dataset, 'id': qid, 'raw_question': item['question'].strip(), 'question': question_with_options, 'options': opts, 'answer': item['answer'], 'answers': [item['answer'], opts[item['answer']]]})
    return rows

@torch.inference_mode()
def encode_queries(texts, device, max_length, batch_size=128):
    tok = AutoTokenizer.from_pretrained(QUERY)
    dtype = torch.float16 if device == 'cuda' else torch.float32
    model = AutoModel.from_pretrained(QUERY, torch_dtype=dtype).to(device).eval()
    out = []
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)
        out.append(model(**enc).last_hidden_state[:, 0, :])
    return torch.cat(out)

def shards(emb_root, corpus_root):
    for corpus in CORPORA:
        for npy in sorted(glob.glob(os.path.join(emb_root, corpus, '*.npy'))):
            name = os.path.basename(npy)[:-4]
            yield (corpus, os.path.join(corpus_root, corpus, 'chunk', name + '.jsonl'), npy)

@torch.inference_mode()
def search(q, emb_root, corpus_root, k, device):
    best_s = torch.full((q.shape[0], k), -1000000000.0, device=device)
    best_i = torch.full((q.shape[0], k), -1, dtype=torch.long, device=device)
    table = []
    for (corpus, jsonl, npy) in shards(emb_root, corpus_root):
        e = torch.from_numpy(np.load(npy)).to(device=device, dtype=q.dtype)
        s = (q @ e.T).float()
        (top_s, top_r) = s.topk(min(k, e.shape[0]), dim=1)
        sid = len(table)
        table.append((corpus, jsonl))
        cand_s = torch.cat([best_s, top_s], 1)
        cand_i = torch.cat([best_i, sid * 10000000 + top_r], 1)
        keep = cand_s.topk(k, dim=1).indices
        (best_s, best_i) = (cand_s.gather(1, keep), cand_i.gather(1, keep))
    return (best_s.cpu().numpy(), best_i.cpu().numpy(), table)

def fetch(best_i, table):
    need = defaultdict(set)
    for row in best_i:
        for code in row:
            need[int(code) // 10000000].add(int(code) % 10000000)
    text = {}
    for (sid, rows) in need.items():
        (corpus, path) = table[sid]
        with open(path, encoding='utf-8') as f:
            for (r, line) in enumerate(f):
                if r in rows:
                    d = json.loads(line)
                    text[sid * 10000000 + r] = {'id': d['id'], 'title': d['title'], 'text': d['content'], 'corpus': corpus}
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', required=True)
    ap.add_argument('--emb-root', required=True)
    ap.add_argument('--corpus-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--query-max-length', type=int, default=512)
    args = ap.parse_args()
    rows = mirage_rows(args.benchmark)[:args.limit]
    q = encode_queries([r['raw_question'] for r in rows], args.device, args.query_max_length)
    (scores, codes, table) = search(q, args.emb_root, args.corpus_root, args.k, args.device)
    text = fetch(codes, table)
    for (r, s_row, c_row) in zip(rows, scores, codes):
        r['retrieved_docs'] = [dict(text[int(c)], score=float(s)) for (s, c) in zip(s_row, c_row)]
    meta = {'retriever': 'MedCPT', 'corpora': list(CORPORA), 'k': args.k, 'query': 'question without options', 'query_max_length': args.query_max_length, 'n': len(rows)}
    json.dump({'meta': meta, 'data': rows}, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False)
    print('RETRIEVAL DONE', len(rows), args.out)
if __name__ == '__main__':
    main()
