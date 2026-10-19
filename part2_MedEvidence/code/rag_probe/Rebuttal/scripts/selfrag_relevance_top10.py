from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / 'scripts'), str(PROJECT_ROOT / 'Rebuttal/scripts'), str(PROJECT_ROOT)]
from step8_selfrag_baseline import build_llm, score_all
from top10_score_contexts import attach_scores, dump_json, flatten_rows, load_topk
from transformers import AutoTokenizer
DEFAULT_MODEL = '/home/qluai/.cache/huggingface/hub/models--selfrag--selfrag_llama2_7b/snapshots/190261383b0779ff66d2f95a73c7ad267d94b820'

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--selector-name', default='selfrag7b')
    p.add_argument('--cache-dir', type=Path, default=PROJECT_ROOT / 'Rebuttal/results/popqa2000_generalization/selfrag_cache')
    p.add_argument('--chunk-size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--tensor-parallel-size', type=int, default=2)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    p.add_argument('--max-model-len', type=int, default=512)
    p.add_argument('--max-num-seqs', type=int, default=128)
    p.add_argument('--max-num-batched-tokens', type=int, default=65536)
    p.add_argument('--dtype', default='float16')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def fingerprint(records: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in records:
        h.update(json.dumps([row['question'], row['doc_text'], row['qidx'], row['rank']], ensure_ascii=False, sort_keys=True).encode())
        h.update(b'\n')
    return h.hexdigest()

def main() -> None:
    args = parse_args()
    if args.output.exists() and (not args.overwrite):
        print(f'[skip completed] {args.output}')
        return
    if args.chunk_size <= 0:
        raise SystemExit('--chunk-size must be positive')
    (meta, rows) = load_topk(args.input, None)
    flat = flatten_rows(rows)
    print(f'[Self-RAG relevance] questions={len(rows)} docs={len(flat)}')
    cfg = SimpleNamespace(model_name=args.model, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens, tensor_parallel_size=args.tensor_parallel_size, enforce_eager=True, disable_enforce_eager=False, trust_remote_code=True, cpu_offload_gb=0.0, max_tokens=1, logprobs=20, truncate_passage=True, batch_size=args.batch_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    llm = build_llm(cfg)
    scores: list[float] = []
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(flat), args.chunk_size):
        end = min(start + args.chunk_size, len(flat))
        chunk = flat[start:end]
        path = args.cache_dir / f'popqa_selfrag_{start:06d}_{end:06d}.json'
        fp = fingerprint(chunk)
        values = None
        if path.exists() and (not args.overwrite):
            obj = json.loads(path.read_text(encoding='utf-8'))
            if obj.get('fingerprint') == fp and len(obj.get('scores', [])) == len(chunk):
                values = [float(x) for x in obj['scores']]
                print(f'[reuse] {path.name}')
        if values is None:
            questions = [x['question'] for x in chunk]
            passages = [x['doc_text'] for x in chunk]
            values = score_all(questions, passages, llm, tokenizer, cfg, split_name=f'{start}:{end}').tolist()
            dump_json(path, {'fingerprint': fp, 'model': args.model, 'scores': values})
            print(f'[saved] {path.name}')
        scores.extend(values)
    scored = attach_scores(rows, flat, scores, {})
    dump_json(args.output, {'meta': {'input': str(args.input), 'dataset': 'popqa', 'selector': args.selector_name, 'model': args.model, 'num_questions': len(rows), 'num_docs': len(flat), 'protocol': 'Self-RAG [Relevant] minus [Irrelevant] first-token log-probability.', 'source_meta': meta}, 'data': scored})
    print(f'[saved final] {args.output}')
if __name__ == '__main__':
    main()
