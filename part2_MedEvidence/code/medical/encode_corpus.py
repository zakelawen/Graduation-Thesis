import argparse
import glob
import json
import os
import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
ARTICLE = 'ncbi/MedCPT-Article-Encoder'

def load_snippets(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append((r['title'], r['content']))
    return rows

@torch.inference_mode()
def encode(model, tokenizer, rows, batch_size, max_length):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]) + len(rows[i][1]))
    batches = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]

    def collate(items):
        idx = items[0]
        return (idx, tokenizer([list(rows[i]) for i in idx], truncation=True, padding=True, max_length=max_length, return_tensors='pt'))
    loader = torch.utils.data.DataLoader(batches, batch_size=1, collate_fn=collate, num_workers=4, prefetch_factor=4)
    out = np.zeros((len(rows), model.config.hidden_size), dtype=np.float16)
    for (idx, enc) in loader:
        enc = {k: v.cuda(non_blocking=True) for (k, v) in enc.items()}
        out[idx] = model(**enc).last_hidden_state[:, 0, :].cpu().numpy()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--max-length', type=int, default=512)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(ARTICLE)
    model = AutoModel.from_pretrained(ARTICLE, attn_implementation='sdpa', torch_dtype=torch.float16).cuda().eval()
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.corpus_dir, '*.jsonl')))
    done = 0
    t0 = time.time()
    for (n, path) in enumerate(files, 1):
        name = os.path.basename(path)[:-6]
        out = os.path.join(args.out_dir, name + '.npy')
        if os.path.exists(out):
            continue
        rows = load_snippets(path)
        emb = encode(model, tokenizer, rows, args.batch_size, args.max_length)
        np.save(out + '.tmp.npy', emb)
        os.replace(out + '.tmp.npy', out)
        done += len(rows)
        print(f'[{n}/{len(files)}] {name} {len(rows)} snippets | {done / (time.time() - t0):.0f} snippets/s', flush=True)
    print('ENCODE DONE', args.corpus_dir, flush=True)
if __name__ == '__main__':
    main()
