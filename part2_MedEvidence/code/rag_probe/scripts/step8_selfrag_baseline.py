import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import gc
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, accuracy_score
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVED_DATA_DIR = PROJECT_ROOT / 'data' / 'final_retrieved'
GOLD_DISTRACTOR_DATA_DIR = PROJECT_ROOT / 'data' / 'final'
RETRIEVED_RESULTS_DIR = PROJECT_ROOT / 'results' / 'probe_retrieved'
GOLD_DISTRACTOR_RESULTS_DIR = PROJECT_ROOT / 'results' / 'probe'
DEFAULT_MODEL_NAME = '/root/autodl-tmp/selfrag/selfrag_llama2_13b'

def parse_args():
    parser = argparse.ArgumentParser(description='Self-RAG baseline for retrieved-document relevance classification.')
    parser.add_argument('--mode', choices=['retrieved', 'gold_distractor'], default='retrieved', help='retrieved = data/final_retrieved; gold_distractor = data/final expanded to rel/dis pairs.')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME, help='HF model name or local model path.')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing train.json and eval.json. Default depends on --mode.')
    parser.add_argument('--results_dir', type=str, default=None, help='Directory to save result JSON.')
    parser.add_argument('--output_name', type=str, default=None, help='Output result JSON filename.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for vLLM generate. Use 1 for 13B on 32GB GPU.')
    parser.add_argument('--threshold', type=float, default=None, help='Fixed threshold. If None, search best threshold on train set.')
    parser.add_argument('--logprobs', type=int, default=20, help='Number of top logprobs returned by vLLM. 20 is safer across vLLM versions. Larger values may require max_logprobs support.')
    parser.add_argument('--max_tokens', type=int, default=1, help='Only generate one token to compare [Relevant] vs [Irrelevant].')
    parser.add_argument('--dtype', type=str, default='float16', choices=['half', 'float16', 'bfloat16', 'float32', 'auto'], help='Model dtype.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8, help='Fraction of GPU memory vLLM can use. Lower this if KV cache initialization OOMs.')
    parser.add_argument('--max_model_len', type=int, default=512, help='Maximum model context length for this baseline.')
    parser.add_argument('--max_num_seqs', type=int, default=1, help='Maximum number of sequences scheduled by vLLM at once.')
    parser.add_argument('--max_num_batched_tokens', type=int, default=512, help='Maximum number of batched tokens. Keep small for 13B on 32GB GPU.')
    parser.add_argument('--cpu_offload_gb', type=float, default=4.0, help='CPU offload size in GB. Use 0 to disable. 4 is recommended for Self-RAG 13B on 32GB GPU.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Use 2 if you want to split the 13B model across two GPUs.')
    parser.add_argument('--enforce_eager', action='store_true', default=True, help='Disable CUDA graph to reduce extra memory usage.')
    parser.add_argument('--disable_enforce_eager', action='store_true', help='If set, use vLLM default CUDA graph behavior.')
    parser.add_argument('--trust_remote_code', action='store_true', default=True, help='Allow loading custom model/tokenizer code when needed.')
    parser.add_argument('--truncate_passage', action='store_true', default=True, help='Manually truncate passage so prompt length fits max_model_len.')
    parser.add_argument('--limit_train', type=int, default=None, help='Optional limit for train samples, useful for smoke test.')
    parser.add_argument('--limit_eval', type=int, default=None, help='Optional limit for eval samples, useful for smoke test.')
    return parser.parse_args()

def extract_doc_text(doc):
    if isinstance(doc, dict):
        title = str(doc.get('title', '')).strip()
        text = str(doc.get('text', '')).strip()
        if title and text:
            return f'Title: {title}\nText: {text}'
        return text or title
    return str(doc).strip()

def load_retrieved_data(split: str, data_dir: Path, limit=None):
    path = data_dir / f'{split}.json'
    if not path.exists():
        raise FileNotFoundError(f'Cannot find data file: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    if limit is not None:
        items = items[:limit]
    questions = []
    passages = []
    labels = []
    sources = []
    for (idx, item) in enumerate(items):
        if 'question' not in item:
            raise KeyError(f'Missing question in item {idx} of {path}')
        if 'doc' not in item:
            raise KeyError(f'Missing doc in item {idx} of {path}')
        if 'label' not in item:
            raise KeyError(f'Missing label in item {idx} of {path}')
        questions.append(str(item['question']))
        passages.append(extract_doc_text(item['doc']))
        labels.append(int(item['label']))
        sources.append(str(item.get('source_dataset', 'unknown')))
    labels = np.array(labels, dtype=np.int64)
    return (questions, passages, labels, sources)

def load_gold_distractor_data(split: str, data_dir: Path, limit=None):
    path = data_dir / f'{split}.json'
    if not path.exists():
        raise FileNotFoundError(f'Cannot find data file: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    if limit is not None:
        items = items[:limit]
    questions = []
    passages = []
    labels = []
    sources = []
    for (idx, item) in enumerate(items):
        if 'question' not in item:
            raise KeyError(f'Missing question in item {idx} of {path}')
        if 'relevant_doc' not in item:
            raise KeyError(f'Missing relevant_doc in item {idx} of {path}')
        if 'distracting_doc' not in item:
            raise KeyError(f'Missing distracting_doc in item {idx} of {path}')
        question = str(item['question'])
        source = str(item.get('source_dataset', 'unknown'))
        questions.append(question)
        passages.append(extract_doc_text(item['relevant_doc']))
        labels.append(1)
        sources.append(source)
        questions.append(question)
        passages.append(extract_doc_text(item['distracting_doc']))
        labels.append(0)
        sources.append(source)
    labels = np.array(labels, dtype=np.int64)
    return (questions, passages, labels, sources)

def load_data(split: str, data_dir: Path, mode: str, limit=None):
    if mode == 'retrieved':
        return load_retrieved_data(split, data_dir, limit)
    if mode == 'gold_distractor':
        return load_gold_distractor_data(split, data_dir, limit)
    raise ValueError(f'Unknown mode: {mode}')

def get_single_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is not None:
        unk_id = getattr(tokenizer, 'unk_token_id', None)
        if unk_id is None or token_id != unk_id:
            return int(token_id)
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if len(encoded) != 1:
        raise ValueError(f'{token} is not a single token under this tokenizer. Encoded ids: {encoded}. This Self-RAG logprob baseline expects single-token special labels.')
    return int(encoded[0])

def format_prompt(question: str, passage: str) -> str:
    return f'### Instruction:\n{question}\n\n### Response:\n[Retrieval]<paragraph>{passage}</paragraph>'

def format_prompt_truncated(question: str, passage: str, tokenizer, max_prompt_tokens: int) -> str:
    prefix = f'### Instruction:\n{question}\n\n### Response:\n[Retrieval]<paragraph>'
    suffix = '</paragraph>'
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    passage_ids = tokenizer.encode(passage, add_special_tokens=False)
    passage_budget = max_prompt_tokens - len(prefix_ids) - len(suffix_ids)
    if passage_budget <= 0:
        full_prompt = format_prompt(question, passage)
        full_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
        full_ids = full_ids[:max_prompt_tokens]
        return tokenizer.decode(full_ids, skip_special_tokens=False)
    if len(passage_ids) > passage_budget:
        passage_ids = passage_ids[:passage_budget]
    prompt_ids = prefix_ids + passage_ids + suffix_ids
    return tokenizer.decode(prompt_ids, skip_special_tokens=False)

def build_sampling_params(args):
    kwargs = {'temperature': 0.0, 'top_p': 1.0, 'max_tokens': args.max_tokens, 'logprobs': args.logprobs}
    kwargs['truncate_prompt_tokens'] = max(1, args.max_model_len - args.max_tokens)
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop('truncate_prompt_tokens', None)
        return SamplingParams(**kwargs)

def build_llm(args):
    enforce_eager = args.enforce_eager and (not args.disable_enforce_eager)
    llm_kwargs = {'model': args.model_name, 'dtype': args.dtype, 'gpu_memory_utilization': args.gpu_memory_utilization, 'max_model_len': args.max_model_len, 'max_num_seqs': args.max_num_seqs, 'max_num_batched_tokens': args.max_num_batched_tokens, 'tensor_parallel_size': args.tensor_parallel_size, 'enforce_eager': enforce_eager, 'trust_remote_code': args.trust_remote_code}
    if args.cpu_offload_gb is not None and args.cpu_offload_gb > 0:
        llm_kwargs['cpu_offload_gb'] = args.cpu_offload_gb
    print('\n=== vLLM Load Config ===')
    for (k, v) in llm_kwargs.items():
        print(f'  {k}: {v}')
    print('========================\n')
    return LLM(**llm_kwargs)

def extract_logprob(logprobs_dict, token_id: int):
    if logprobs_dict is None:
        return None
    obj = logprobs_dict.get(token_id, None)
    if obj is None:
        return None
    if hasattr(obj, 'logprob'):
        return float(obj.logprob)
    if isinstance(obj, dict) and 'logprob' in obj:
        return float(obj['logprob'])
    return None

def score_all(questions, passages, model, tokenizer, args, split_name='data'):
    relevant_id = get_single_token_id(tokenizer, '[Relevant]')
    irrelevant_id = get_single_token_id(tokenizer, '[Irrelevant]')
    print(f'  [Relevant] token id: {relevant_id}')
    print(f'  [Irrelevant] token id: {irrelevant_id}')
    sampling_params = build_sampling_params(args)
    max_prompt_tokens = max(1, args.max_model_len - args.max_tokens)
    prompts = []
    for (q, p) in zip(questions, passages):
        if args.truncate_passage:
            prompts.append(format_prompt_truncated(q, p, tokenizer, max_prompt_tokens))
        else:
            prompts.append(format_prompt(q, p))
    all_scores = []
    missing_rel = 0
    missing_irrel = 0
    for i in tqdm(range(0, len(prompts), args.batch_size), desc=f'Scoring {split_name}'):
        batch_prompts = prompts[i:i + args.batch_size]
        outputs = model.generate(batch_prompts, sampling_params)
        for output in outputs:
            if not output.outputs:
                raise RuntimeError('vLLM returned empty output.')
            token_logprobs = output.outputs[0].logprobs
            if token_logprobs is None or len(token_logprobs) == 0:
                raise RuntimeError('No logprobs returned by vLLM. Please check SamplingParams(logprobs=...).')
            logprobs_dict = token_logprobs[0]
            logprob_rel = extract_logprob(logprobs_dict, relevant_id)
            logprob_irrel = extract_logprob(logprobs_dict, irrelevant_id)
            if logprob_rel is None:
                missing_rel += 1
                logprob_rel = -100.0
            if logprob_irrel is None:
                missing_irrel += 1
                logprob_irrel = -100.0
            score = logprob_rel - logprob_irrel
            all_scores.append(score)
    if missing_rel > 0 or missing_irrel > 0:
        print(f'  Warning: in {split_name}, missing top-logprob entries: [Relevant]={missing_rel}/{len(prompts)}, [Irrelevant]={missing_irrel}/{len(prompts)}. If many are missing, try increasing --logprobs if your vLLM supports it.')
    return np.array(all_scores, dtype=np.float32)

def safe_auroc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(set(labels.tolist())) < 2:
        return float('nan')
    return float(roc_auc_score(labels, scores))

def evaluate(scores, labels, sources, threshold=0.0):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    sources = np.asarray(sources)
    datasets = sorted(set(sources.tolist()))
    results = {}
    for subset_name in datasets + ['combined']:
        if subset_name == 'combined':
            mask = np.ones(len(labels), dtype=bool)
        else:
            mask = sources == subset_name
        s = scores[mask]
        l = labels[mask]
        preds = (s > threshold).astype(int)
        auroc = safe_auroc(l, s)
        acc = float(accuracy_score(l, preds))
        margin = float(np.abs(s).mean()) if len(s) > 0 else float('nan')
        results[subset_name] = {'auroc': auroc, 'accuracy': acc, 'avg_margin': margin, 'n_samples': int(mask.sum()), 'n_relevant': int(l.sum()), 'n_distracting': int((1 - l).sum())}
    return results

def find_best_threshold(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if len(scores) == 0:
        raise ValueError('Cannot find threshold from empty scores.')
    unique_scores = np.unique(scores)
    if len(unique_scores) == 1:
        candidates = np.array([unique_scores[0]], dtype=np.float32)
    else:
        mids = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidates = np.concatenate([[unique_scores[0] - 1e-06], mids, [unique_scores[-1] + 1e-06]])
    best_acc = -1.0
    best_t = 0.0
    for t in candidates:
        preds = (scores > t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = float(acc)
            best_t = float(t)
    return (best_t, best_acc)

def print_results(title, results):
    print(f'\n=== {title} ===')
    for (subset, metrics) in results.items():
        auroc = metrics['auroc']
        auroc_str = 'nan' if np.isnan(auroc) else f'{auroc:.4f}'
        print(f"  {subset:12s}: AUROC={auroc_str}  Acc={metrics['accuracy']:.4f}  Margin={metrics['avg_margin']:.4f}  n={metrics['n_samples']}  rel={metrics['n_relevant']}  dist={metrics['n_distracting']}")

def main():
    args = parse_args()
    if args.data_dir is None:
        data_dir = RETRIEVED_DATA_DIR if args.mode == 'retrieved' else GOLD_DISTRACTOR_DATA_DIR
    else:
        data_dir = Path(args.data_dir)
    if args.results_dir is None:
        results_dir = RETRIEVED_RESULTS_DIR if args.mode == 'retrieved' else GOLD_DISTRACTOR_RESULTS_DIR
    else:
        results_dir = Path(args.results_dir)
    if args.output_name is None:
        output_name = 'selfrag_baseline_results.json' if args.mode == 'retrieved' else 'selfrag_gold_distractor_baseline_results.json'
    else:
        output_name = args.output_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f'Project root: {PROJECT_ROOT}')
    print(f'Mode:         {args.mode}')
    print(f'Data dir:     {data_dir}')
    print(f'Results dir:  {results_dir}')
    print(f'\nLoading tokenizer: {args.model_name} ...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    print(f'Loading Self-RAG model: {args.model_name} ...')
    model = build_llm(args)
    print('  Model loaded.')
    print('\nLoading data ...')
    (train_q, train_p, train_labels, train_sources) = load_data('train', data_dir, args.mode, limit=args.limit_train)
    (eval_q, eval_p, eval_labels, eval_sources) = load_data('eval', data_dir, args.mode, limit=args.limit_eval)
    print(f'  Train: {len(train_q)}')
    print(f'  Eval:  {len(eval_q)}')
    print(f'  Train label distribution: rel={int(train_labels.sum())}, dist={int((1 - train_labels).sum())}')
    print(f'  Eval label distribution:  rel={int(eval_labels.sum())}, dist={int((1 - eval_labels).sum())}')
    print('\nScoring train set ...')
    train_scores = score_all(train_q, train_p, model, tokenizer, args, split_name='train')
    print('\nScoring eval set ...')
    eval_scores = score_all(eval_q, eval_p, model, tokenizer, args, split_name='eval')
    if args.threshold is not None:
        threshold = float(args.threshold)
        print(f'\nUsing fixed threshold: {threshold:.6f}')
    else:
        (threshold, train_acc) = find_best_threshold(train_scores, train_labels)
        print(f'\nBest threshold from train set: {threshold:.6f} (train acc: {train_acc:.4f})')
    eval_results = evaluate(eval_scores, eval_labels, eval_sources, threshold)
    train_results = evaluate(train_scores, train_labels, train_sources, threshold)
    print_results('Train Results', train_results)
    print_results('Eval Results', eval_results)
    output = {'method': 'selfrag_retrieval_evaluator', 'mode': args.mode, 'model_name': args.model_name, 'data_dir': str(data_dir), 'threshold': threshold, 'config': {'batch_size': args.batch_size, 'dtype': args.dtype, 'gpu_memory_utilization': args.gpu_memory_utilization, 'max_model_len': args.max_model_len, 'max_num_seqs': args.max_num_seqs, 'max_num_batched_tokens': args.max_num_batched_tokens, 'cpu_offload_gb': args.cpu_offload_gb, 'tensor_parallel_size': args.tensor_parallel_size, 'enforce_eager': args.enforce_eager and (not args.disable_enforce_eager), 'logprobs': args.logprobs, 'max_tokens': args.max_tokens, 'truncate_passage': args.truncate_passage, 'limit_train': args.limit_train, 'limit_eval': args.limit_eval}, 'eval': eval_results, 'train': train_results}
    out_path = results_dir / output_name
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\nSaved results → {out_path}')
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
