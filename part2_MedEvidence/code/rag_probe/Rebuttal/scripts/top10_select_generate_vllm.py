from __future__ import annotations
import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REBUTTAL_ROOT = PROJECT_ROOT / 'Rebuttal'
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
sys.path.insert(0, str(REBUTTAL_ROOT / 'scripts'))
sys.path.insert(0, str(PROJECT_ROOT))
from step11_contrastive_decoding import MODEL_REGISTRY, build_prompt, clean_prediction
from top10_select_generate import build_multi_doc_context, docs_for_row, dump_json, load_rows, output_path, sanitize_name, select_docs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', default='gemma9b', choices=list(MODEL_REGISTRY))
    parser.add_argument('--output-dir', type=Path, default=REBUTTAL_ROOT / 'results/top10_context/generations')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max-context-docs', type=int, default=10)
    parser.add_argument('--max-doc-words', type=int, default=120)
    parser.add_argument('--max-new-tokens', type=int, default=20)
    parser.add_argument('--max-input-length', type=int, default=2048)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--max-num-seqs', type=int, default=256)
    parser.add_argument('--max-num-batched-tokens', type=int, default=None)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def load_manifest(path: Path) -> list[dict[str, Any]]:
    jobs = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Manifest must be a non-empty JSON list')
    required = {'input', 'dataset', 'selector_name', 'rule'}
    for (index, job) in enumerate(jobs):
        missing = required - set(job)
        if missing:
            raise ValueError(f'Job {index} missing fields: {sorted(missing)}')
    return jobs

def job_namespace(global_args: argparse.Namespace, job: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(model=global_args.model, dataset=job['dataset'], selector_name=job['selector_name'], rule=job['rule'], threshold=float(job.get('threshold', 0.5)), score_direction=str(job.get('score_direction', 'greater')), select_top_k=int(job.get('select_top_k', 1)), max_context_docs=global_args.max_context_docs, max_doc_words=global_args.max_doc_words, max_new_tokens=global_args.max_new_tokens, max_input_length=global_args.max_input_length, output_dir=global_args.output_dir, limit=global_args.limit, decoding_method=str(job.get('decoding_method', 'vanilla_rag')), alpha=1.0, temperature=0.0, top_p=1.0, contrast_layer=23)

def prepare_job(global_args: argparse.Namespace, job: dict[str, Any]) -> tuple[SimpleNamespace, Path, list[dict[str, Any]], list[str], list[int]]:
    args = job_namespace(global_args, job)
    gate_name = sanitize_name(args.selector_name)
    out_path = output_path(args, args.dataset, gate_name, args.decoding_method)
    if out_path.exists() and (not global_args.overwrite):
        with out_path.open(encoding='utf-8') as existing_file:
            header = existing_file.read(65536).split('"outputs"', 1)[0]
        if '"backend": "vllm"' in header:
            print(f'[skip completed vllm] {out_path}')
            return (args, out_path, [], [], [])
        print(f'[replace non-vllm output] {out_path}')
    (_, rows) = load_rows(PROJECT_ROOT / job['input'], global_args.limit)
    prepared = []
    prompts = []
    selected_counts = []
    for (idx, row) in enumerate(rows):
        scored_docs = docs_for_row(row)
        selected = select_docs(scored_docs, args)
        context = build_multi_doc_context(selected, args.max_doc_words)
        prompts.append(build_prompt(row['question'], context))
        selected_counts.append(len(selected))
        prepared.append({'idx': row.get('idx', idx), 'row': row, 'scored_docs': scored_docs, 'selected': selected, 'has_context': bool(context)})
    return (args, out_path, prepared, prompts, selected_counts)

def serialize_job(args: SimpleNamespace, input_file: str, out_path: Path, prepared: list[dict[str, Any]], selected_counts: list[int], texts: list[str], elapsed: float) -> None:
    gate_name = sanitize_name(args.selector_name)
    outputs = []
    for (item, raw_text) in zip(prepared, texts):
        row = item['row']
        scored_docs = item['scored_docs']
        selected = item['selected']
        selected_scores = [float(x.get('score', 0.0)) for x in selected]
        available_scores = [float(x.get('score', 0.0)) for x in scored_docs]
        trusted = bool(selected)
        prediction = clean_prediction(raw_text)
        outputs.append({'idx': item['idx'], 'source_dataset': row.get('source_dataset', args.dataset), 'question': row['question'], 'answers': row.get('answers', []), 'retrieved_doc': selected[0]['doc'] if selected else None, 'available_docs': [x.get('doc') for x in scored_docs], 'selected_docs': [x['doc'] for x in selected], 'selection': {'gate': gate_name, 'rule': args.rule, 'threshold': float(args.threshold), 'select_top_k': int(args.select_top_k), 'num_available_docs': len(scored_docs), 'num_selected_docs': len(selected), 'selected_ranks': [int(x.get('rank', 0)) for x in selected], 'selected_scores': selected_scores, 'max_available_score': float(max(available_scores, default=0.0))}, 'csv': {'p_relevant': float(max(selected_scores, default=0.0)), 'pred_label': int(trusted), 'pred_name': 'relevant' if trusted else 'distracting', 'margin': None, 'logits': None, 'trusted_by_tau': trusted, 'tau': float(args.threshold), 'gate_source': gate_name}, 'generation': {'method': f'{gate_name}_gate_{args.decoding_method}', 'gate': gate_name, 'decoding_method': args.decoding_method, 'actual_mode': args.decoding_method if trusted else 'no_doc', 'alpha_requested': 1.0, 'alpha_used': 0.0, 'max_new_tokens': int(args.max_new_tokens), 'temperature': 0.0, 'top_p': 1.0, 'prediction': prediction, 'details': {'backend': 'vllm', 'prompt': 'selected_multi_document_rag' if trusted else 'no_doc', 'max_doc_words': int(args.max_doc_words)}}})
    trusted_count = sum((count > 0 for count in selected_counts))
    avg_ctx = sum(selected_counts) / max(1, len(selected_counts))
    payload = {'meta': {'model': args.model, 'dataset': args.dataset, 'method': f'{gate_name}_gate_{args.decoding_method}', 'selector': gate_name, 'gate_method': gate_name, 'decoding_method': args.decoding_method, 'selection_rule': args.rule, 'threshold': float(args.threshold), 'score_direction': args.score_direction, 'tau': float(args.threshold), 'select_top_k': int(args.select_top_k), 'alpha': 1.0, 'temperature': 0.0, 'top_p': 1.0, 'max_new_tokens': int(args.max_new_tokens), 'max_input_length': args.max_input_length, 'max_context_docs': int(args.max_context_docs), 'max_doc_words': int(args.max_doc_words), 'input_file': input_file, 'num_samples': len(outputs), 'trusted_docs': int(trusted_count), 'trusted_rate': trusted_count / max(1, len(outputs)), 'avg_selected_docs': avg_ctx, 'num_no_context': int(sum((count == 0 for count in selected_counts))), 'elapsed_sec': elapsed, 'backend': 'vllm', 'top10_selection_experiment': True}, 'outputs': outputs}
    dump_json(out_path, payload)
    print(f'[saved] {out_path}; n={len(outputs)}; avg_ctx={avg_ctx:.3f}')

def main() -> None:
    args = parse_args()
    jobs = load_manifest(args.manifest)
    pending: list[tuple[dict[str, Any], SimpleNamespace, Path, list[dict[str, Any]], list[str], list[int]]] = []
    for job in jobs:
        (prepared_args, out_path, prepared, prompts, counts) = prepare_job(args, job)
        if prompts:
            pending.append((job, prepared_args, out_path, prepared, prompts, counts))
    if not pending:
        print('[complete] all manifest jobs already have compatible vLLM outputs')
        return
    print(f'[pending] {len(pending)}/{len(jobs)} manifest jobs require generation')
    conda_libstdcpp = Path(sys.prefix) / 'lib' / 'libstdc++.so.6'
    if conda_libstdcpp.exists():
        conda_lib_dir = str(conda_libstdcpp.parent)
        old_library_path = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f'{conda_lib_dir}:{old_library_path}' if old_library_path else conda_lib_dir
        ctypes.CDLL(str(conda_libstdcpp), mode=ctypes.RTLD_GLOBAL)
    from vllm import LLM, SamplingParams
    model_path = MODEL_REGISTRY[args.model]['hf_name']
    print(f'[vllm] loading {model_path}')
    llm_kwargs = dict(model=model_path, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_input_length + args.max_new_tokens, max_num_seqs=args.max_num_seqs, trust_remote_code=True)
    if args.max_num_batched_tokens is not None:
        llm_kwargs['max_num_batched_tokens'] = args.max_num_batched_tokens
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, stop=['\n'])
    for (index, (job, prepared_args, out_path, prepared, prompts, counts)) in enumerate(pending, start=1):
        print(f"[job {index}/{len(pending)}] {job['dataset']} {job['selector_name']} n={len(prompts)}")
        start = time.time()
        tokenizer = llm.get_tokenizer()
        token_prompts = [{'prompt_token_ids': tokenizer.encode(prompt, add_special_tokens=True)[-args.max_input_length:]} for prompt in prompts]
        request_outputs = llm.generate(token_prompts, sampling, use_tqdm=True)
        texts = [output.outputs[0].text if output.outputs else '' for output in request_outputs]
        serialize_job(prepared_args, job['input'], out_path, prepared, counts, texts, time.time() - start)
if __name__ == '__main__':
    main()
