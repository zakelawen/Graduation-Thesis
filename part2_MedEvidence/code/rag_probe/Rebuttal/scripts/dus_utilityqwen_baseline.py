from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rebuttal_common import REBUTTAL_ROOT, dump_json, load_data, print_eval, select_threshold_and_evaluate
DEFAULT_SELECTOR = '/home/qluai/.cache/huggingface/hub/models--hengranZhang--UtilityQwen1.7B/snapshots/986a2ccbef147749fb29c3728a25ea1e4a1101df/UtilityQwen1.7B-model_checkpoint'
MAX_PASSAGE_WORDS = 300

def load_custom_retrieved(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8') as handle:
        items = json.load(handle)
    if isinstance(items, dict):
        items = items.get('data', items.get('records', []))
    records = []
    for (idx, item) in enumerate(items):
        doc = item.get('doc', item.get('retrieved_doc', {}))
        if isinstance(doc, dict):
            title = str(doc.get('title', '')).strip()
            text = str(doc.get('text', '')).strip()
            doc_text = f'Title: {title}\nText: {text}' if title else text
        else:
            doc_text = str(doc)
        records.append({'id': item.get('id', idx), 'question': item['question'], 'doc_text': doc_text, 'answers': item.get('answers', []), 'label': int(item['label']), 'source': str(item.get('source_dataset', item.get('source', 'unknown'))).lower().strip()})
    return records if limit is None else records[:limit]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', default='retrieved', choices=('retrieved', 'gold_distractor'))
    p.add_argument('--train-path', type=Path, default=None, help='Optional custom training JSON.')
    p.add_argument('--eval-path', type=Path, default=None, help='Optional custom evaluation JSON.')
    p.add_argument('--selector-path', default=os.environ.get('UTILITYQWEN_PATH', DEFAULT_SELECTOR))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-samples', type=int, default=5)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--max-len', type=int, default=2048)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--output-dir', type=Path, default=REBUTTAL_ROOT / 'results' / 'dus_utilityqwen')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def prefix(query: str, num: int) -> list[dict[str, str]]:
    return [{'role': 'system', 'content': 'You are the utility selector, an intelligent assistant that can select the passages that have utility in answering the question.'}, {'role': 'assistant', 'content': 'Yes, i am the utility selector.'}, {'role': 'user', 'content': f'I will provide you with {num} passages, each indicated by number identifier []. \nSelect the passages that have utility in answering the question: {query}.'}, {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

def post(query: str, num: int) -> str:
    return f'Question: {query}\nUtility Requirement: A passage has **utility** only if it is **both relevant to the question AND useful for generating a correct and reasonable answer.**\nFirstly, **provide the answer** to the question based on the provided {num} passages or your own knowledge without sources, which can help judge passages utility.\nThen, **select the passages that have utility in answering the question**.\n**Output Format STRICTLY:**\nAnswer: [Your generated answer here]\nMy selection: [[i],[j],...]\nOnly respond to the answer and the selected list, do not say any word or explain.'

def truncate_words(text: str, max_words: int=MAX_PASSAGE_WORDS) -> str:
    return ' '.join(text.replace('Title: Content: ', '').split()[:max_words])

def build_messages(question: str, doc_text: str) -> list[dict[str, str]]:
    messages = prefix(question, 1)
    messages.append({'role': 'user', 'content': f'[1] {truncate_words(doc_text)}'})
    messages.append({'role': 'assistant', 'content': 'Received passage [1].'})
    messages.append({'role': 'user', 'content': post(question, 1)})
    return messages

def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_selected_one(text: str) -> int:
    lower = (text or '').lower()
    segment = lower.split('selection:', 1)[1] if 'selection:' in lower else lower
    indices = [int(x) for x in re.findall('\\[(\\d+)\\]', segment)]
    return int(1 in indices)

def load_selector(selector_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(selector_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(selector_path, torch_dtype=torch.bfloat16).to(device).eval()
    return (tokenizer, model)

@torch.no_grad()
def score_records(records: list[dict[str, Any]], tokenizer: Any, model: Any, device: torch.device, args: argparse.Namespace) -> list[float]:
    prompts = [apply_chat_template(tokenizer, build_messages(r['question'], r['doc_text'])) for r in records]
    scores: list[float] = []
    do_sample = args.temperature > 0.0 or args.num_samples > 1
    for start in tqdm(range(0, len(prompts), args.batch_size), desc='DUS scoring'):
        batch = prompts[start:start + args.batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len).to(device)
        generated = model.generate(**enc, do_sample=do_sample, temperature=args.temperature if do_sample else None, top_p=0.9 if do_sample else None, num_return_sequences=args.num_samples, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        generated = generated[:, enc['input_ids'].shape[1]:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for offset in range(len(batch)):
            outs = texts[offset * args.num_samples:(offset + 1) * args.num_samples]
            votes = [parse_selected_one(x) for x in outs]
            scores.append(float(np.mean(votes)) if votes else 0.0)
    return scores

def score_cache_path(args: argparse.Namespace, split: str) -> Path:
    return args.output_dir / f'{args.mode}_{split}_utilityqwen_samples{args.num_samples}.scores.json'

def compute_or_load_split(split: str, tokenizer: Any, model: Any, device: torch.device, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[float]]:
    custom_path = args.train_path if split == 'train' else args.eval_path
    if custom_path is not None:
        if args.mode != 'retrieved':
            raise ValueError('Custom paths currently require --mode retrieved')
        records = load_custom_retrieved(custom_path, args.limit)
    else:
        records = load_data(split, args.mode, args.limit)
    path = score_cache_path(args, split)
    if path.exists() and (not args.overwrite):
        with path.open() as f:
            obj = json.load(f)
        scores = [float(x) for x in obj['scores']]
        if len(scores) != len(records):
            raise SystemExit(f'cached score count does not match records: {path}')
        print(f'[DUS] loading cached {split} scores: {path}')
        return (records, scores)
    print(f'[DUS] scoring {split}: {len(records)} records')
    scores = score_records(records, tokenizer, model, device, args)
    dump_json(path, {'method': 'dus_utilityqwen_pointwise', 'split': split, 'mode': args.mode, 'scores': scores, 'labels': [int(r['label']) for r in records], 'sources': [r['source'] for r in records], 'config': {'selector_path': args.selector_path, 'num_samples': args.num_samples, 'temperature': args.temperature, 'max_new_tokens': args.max_new_tokens, 'limit': args.limit, 'train_path': str(args.train_path) if args.train_path else None, 'eval_path': str(args.eval_path) if args.eval_path else None}})
    print(f'[DUS] saved {split} scores -> {path}')
    return (records, scores)

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[DUS] selector={args.selector_path}')
    print(f'[DUS] mode={args.mode} device={device} samples={args.num_samples}')
    (tokenizer, model) = load_selector(args.selector_path, device)
    (train_records, train_scores) = compute_or_load_split('train', tokenizer, model, device, args)
    (eval_records, eval_scores) = compute_or_load_split('eval', tokenizer, model, device, args)
    result = select_threshold_and_evaluate(train_scores, [int(r['label']) for r in train_records], eval_scores, [int(r['label']) for r in eval_records], [r['source'] for r in eval_records], train_sources=[r['source'] for r in train_records])
    print('\n=== DUS / UtilityQwen Eval Results ===')
    print_eval(result['eval'])
    summary_path = args.output_dir / f'{args.mode}_utilityqwen_samples{args.num_samples}_summary.json'
    dump_json(summary_path, {'method': 'dus_utilityqwen_pointwise', 'note': 'Answer-free pointwise adaptation of the DUS listwise utility selector.', 'mode': args.mode, 'selector_path': args.selector_path, 'config': {'num_samples': args.num_samples, 'temperature': args.temperature, 'max_new_tokens': args.max_new_tokens, 'batch_size': args.batch_size, 'limit': args.limit, 'train_path': str(args.train_path) if args.train_path else None, 'eval_path': str(args.eval_path) if args.eval_path else None}, **result})
    print(f'\nSaved summary -> {summary_path}')
if __name__ == '__main__':
    main()
