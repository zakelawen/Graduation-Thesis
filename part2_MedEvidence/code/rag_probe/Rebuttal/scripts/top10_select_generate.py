from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from tqdm import tqdm
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REBUTTAL_ROOT = PROJECT_ROOT / 'Rebuttal'
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
sys.path.insert(0, str(REBUTTAL_ROOT / 'scripts'))
sys.path.insert(0, str(PROJECT_ROOT))
from rebuttal_common import extract_doc_text
from step11_contrastive_decoding import DATASETS, MODEL_REGISTRY, build_prompt, clean_prediction, current_device, format_float_for_name, generate_acd, generate_cad, generate_context_ucd, generate_dola, generate_layer_delta_cad, generate_standard, load_base_model_and_tokenizer, parse_dola_layers_config
SELECTION_RULES = ('none', 'top10', 'top1', 'score_threshold', 'score_topk', 'score_threshold_or_top1')
DECODING_METHODS = ('no_doc', 'standard_rag', 'vanilla_rag', 'cad_fixed', 'cad', 'acd', 'context_ucd', 'dola', 'layer_delta_cad', 'replug')

def load_json(path: Path) -> Any:
    with path.open(encoding='utf-8') as f:
        return json.load(f)

def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, required=True, help='Raw top-k file or scored file from top10_score_contexts.py')
    p.add_argument('--model', required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument('--dataset', default=None, choices=DATASETS)
    p.add_argument('--selector-name', '--gate-name', dest='selector_name', default=None, help='Gate/selector name recorded in output meta/table')
    p.add_argument('--rule', required=True, choices=SELECTION_RULES, help='How selected contexts are chosen from top-k/scored docs')
    p.add_argument('--threshold', type=float, default=0.5, help='Passage relevance threshold for score_threshold rules')
    p.add_argument('--score-direction', choices=('greater', 'less'), default='greater')
    p.add_argument('--select-top-k', type=int, default=1, help='For score_topk')
    p.add_argument('--max-context-docs', type=int, default=10)
    p.add_argument('--max-doc-words', type=int, default=120)
    p.add_argument('--decoding-method', default='vanilla_rag', choices=DECODING_METHODS)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--output-dir', type=Path, default=REBUTTAL_ROOT / 'results' / 'top10_context' / 'generations')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--alpha', type=float, default=1.0, help='CAD/ACD/layer-delta strength')
    p.add_argument('--contrast-layer', type=int, default=23, help='Step10 hidden_states index for layer_delta_cad')
    p.add_argument('--context-ucd-beta', type=float, default=1.0)
    p.add_argument('--context-ucd-relative-top', type=float, default=0.0)
    p.add_argument('--dola-layers', type=str, default=None)
    p.add_argument('--dola-repetition-penalty', type=float, default=1.2)
    p.add_argument('--dola-mature-layer', type=int, default=-1)
    p.add_argument('--dola-premature-layer', type=int, default=None)
    p.add_argument('--dola-candidate-layers', type=str, default=None)
    p.add_argument('--replug-fusion', choices=('flashrag_logits', 'probability'), default='flashrag_logits')
    p.add_argument('--replug-weighting', choices=('flashrag', 'softmax'), default='flashrag')
    p.add_argument('--replug-score-temperature', type=float, default=1.0, help='Softmax temperature used only when --replug-weighting softmax')
    p.add_argument('--max-new-tokens', type=int, default=20)
    p.add_argument('--max-input-length', type=int, default=2048)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--stop-on-newline', action='store_true', default=True)
    p.add_argument('--no-stop-on-newline', action='store_false', dest='stop_on_newline')
    return p.parse_args()

def tokenize_prompts_left_padded(tokenizer: Any, prompts: list[str], device: torch.device, max_input_length: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError('Tokenizer must define pad_token_id or eos_token_id')
    encoded = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids[0]
        if max_input_length is not None and ids.numel() > max_input_length:
            ids = ids[-max_input_length:]
        encoded.append(ids)
    max_len = max((int(x.numel()) for x in encoded))
    input_ids = torch.full((len(encoded), max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
    for (row_idx, ids) in enumerate(encoded):
        n = int(ids.numel())
        input_ids[row_idx, -n:] = ids
        attention_mask[row_idx, -n:] = 1
    return (input_ids.to(device), attention_mask.to(device))

def replug_weights(scores: list[float], temperature: float, weighting: str, device: torch.device) -> torch.Tensor:
    values = torch.tensor(scores, dtype=torch.float32, device=device)
    if not torch.isfinite(values).all():
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if weighting == 'flashrag':
        denom = values.sum()
        if torch.abs(denom).item() > 1e-12:
            return values / denom
        return torch.full_like(values, 1.0 / max(1, values.numel()))
    if weighting == 'softmax':
        if temperature <= 0:
            raise ValueError('--replug-score-temperature must be positive')
        return torch.softmax((values - values.max()) / float(temperature), dim=0)
    raise ValueError(f'Bad REPLUG weighting: {weighting}')

@torch.no_grad()
def generate_replug(model: Any, tokenizer: Any, prompts: list[str], scores: list[float], max_new_tokens: int, max_input_length: int | None, temperature: float, top_p: float, stop_on_newline: bool, score_temperature: float, fusion: str, weighting: str) -> tuple[str, dict[str, Any]]:
    if temperature is not None and temperature > 0:
        raise ValueError('REPLUG baseline is implemented for greedy decoding only.')
    if top_p is not None and top_p < 1.0:
        raise ValueError('REPLUG baseline currently expects --top-p 1.0 for controlled greedy decoding.')
    if not prompts:
        raise ValueError('REPLUG requires at least one document prompt')
    if len(prompts) != len(scores):
        raise ValueError('REPLUG prompts and scores must have the same length')
    device = current_device(model)
    (input_ids, attention_mask) = tokenize_prompts_left_padded(tokenizer, prompts, device, max_input_length)
    weights = replug_weights(scores, score_temperature, weighting=weighting, device=device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    logits = out.logits[:, -1, :]
    past = out.past_key_values
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id
    num_docs = len(prompts)
    for step_idx in range(max_new_tokens):
        if fusion == 'flashrag_logits':
            fused_scores = (weights.unsqueeze(-1) * logits.float()).sum(dim=0, keepdim=True)
        elif fusion == 'probability':
            log_probs = F.log_softmax(logits.float(), dim=-1)
            fused_scores = torch.logsumexp(log_probs + weights.log().unsqueeze(-1), dim=0, keepdim=True)
        else:
            raise ValueError(f'Bad REPLUG fusion: {fusion}')
        next_id = torch.argmax(fused_scores, dim=-1, keepdim=True)
        token_id = int(next_id.item())
        if eos_id is not None and token_id == eos_id:
            break
        generated.append(token_id)
        if stop_on_newline:
            text = tokenizer.decode(generated, skip_special_tokens=True)
            if '\n' in text:
                break
        if step_idx == max_new_tokens - 1:
            break
        next_ids = next_id.repeat(num_docs, 1)
        attention_mask = torch.cat([attention_mask, torch.ones((num_docs, 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
        out = model(input_ids=next_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
    text = tokenizer.decode(generated, skip_special_tokens=True)
    details = {'implementation': 'replug_flashrag_compatible_weighted_ensemble', 'fusion': fusion, 'weighting': weighting, 'num_replug_docs': num_docs, 'score_temperature': float(score_temperature), 'weights': [float(x) for x in weights.detach().cpu().tolist()], 'raw_scores': [float(x) for x in scores]}
    return (clean_prediction(text), details)

def sanitize_name(text: str) -> str:
    text = text.strip().replace('-', '_')
    if not re.fullmatch('[A-Za-z0-9_]+', text):
        raise ValueError(f'Bad name: {text!r}')
    return text

def normalize_decoding_method(method: str) -> str:
    if method == 'standard_rag':
        return 'vanilla_rag'
    if method == 'cad':
        return 'cad_fixed'
    return method

def load_rows(path: Path, limit: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = load_json(path)
    if isinstance(obj, dict) and 'data' in obj:
        meta = obj.get('meta', {}) if isinstance(obj.get('meta'), dict) else {}
        rows = obj['data']
    elif isinstance(obj, list):
        meta = {}
        rows = obj
    else:
        raise ValueError(f'Bad input format: {path}')
    if limit is not None:
        rows = rows[:limit]
    return (meta, rows)

def docs_for_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    if row.get('scored_docs'):
        return row['scored_docs']
    docs = row.get('retrieved_docs') or []
    if not docs and row.get('retrieved_doc') is not None:
        docs = [row['retrieved_doc']]
    return [{'rank': i, 'score': float((doc or {}).get('score', 0.0)), 'doc': doc, 'detail': {}} for (i, doc) in enumerate(docs)]

def select_docs(scored_docs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    docs = list(scored_docs)
    if args.rule == 'none':
        selected = []
    elif args.rule == 'top10':
        selected = docs[:args.max_context_docs]
    elif args.rule == 'top1':
        selected = docs[:1]
    elif args.rule == 'score_threshold':
        if getattr(args, 'score_direction', 'greater') == 'less':
            selected = [x for x in docs if float(x.get('score', 0.0)) <= args.threshold]
        else:
            selected = [x for x in docs if float(x.get('score', 0.0)) >= args.threshold]
    elif args.rule == 'score_threshold_or_top1':
        direction = getattr(args, 'score_direction', 'greater')
        if direction == 'less':
            selected = [x for x in docs if float(x.get('score', 0.0)) <= args.threshold]
        else:
            selected = [x for x in docs if float(x.get('score', 0.0)) >= args.threshold]
        if not selected and docs:
            key = lambda x: float(x.get('score', 0.0))
            selected = [min(docs, key=key) if direction == 'less' else max(docs, key=key)]
    elif args.rule == 'score_topk':
        reverse = getattr(args, 'score_direction', 'greater') != 'less'
        selected = sorted(docs, key=lambda x: float(x.get('score', 0.0)), reverse=reverse)[:args.select_top_k]
    else:
        raise ValueError(args.rule)
    selected = sorted(selected, key=lambda x: int(x.get('rank', 0)))
    return selected[:args.max_context_docs]

def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if max_words is not None and max_words > 0 and (len(words) > max_words):
        return ' '.join(words[:max_words])
    return text

def build_multi_doc_context(selected: list[dict[str, Any]], max_doc_words: int) -> str | None:
    if not selected:
        return None
    chunks = []
    for (local_idx, item) in enumerate(selected, start=1):
        doc_text = truncate_words(extract_doc_text(item['doc']), max_doc_words)
        chunks.append(f'[{local_idx}] {doc_text}')
    return '\n\n'.join(chunks)

def output_path(args: argparse.Namespace, dataset: str, gate_name: str, decoding_method: str) -> Path:
    thr = format_float_for_name(args.threshold)
    temp = format_float_for_name(args.temperature)
    alpha = format_float_for_name(args.alpha)
    limit = f'_limit{args.limit}' if args.limit is not None else ''
    topk = f'_k{args.select_top_k}' if args.rule == 'score_topk' else ''
    layer = f'_layer{args.contrast_layer}' if decoding_method == 'layer_delta_cad' else ''
    return args.output_dir / f'{args.model}_{dataset}_{gate_name}_{args.rule}{topk}_{decoding_method}_alpha{alpha}_thr{thr}_temp{temp}{layer}{limit}.json'

def generate_with_backend(model: Any, tokenizer: Any, method: str, doc_prompt: str, no_prompt: str, has_context: bool, selected: list[dict[str, Any]], question: str, args: argparse.Namespace) -> tuple[str, str, float, dict[str, Any]]:
    if method == 'no_doc' or not has_context:
        prediction = generate_standard(model=model, tokenizer=tokenizer, prompt=no_prompt, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
        return (prediction, 'no_doc', 0.0, {})
    if method == 'vanilla_rag':
        prediction = generate_standard(model=model, tokenizer=tokenizer, prompt=doc_prompt, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
        return (prediction, 'vanilla_rag', 0.0, {})
    if method == 'replug':
        prompts = []
        scores = []
        for item in selected:
            doc_text = truncate_words(extract_doc_text(item['doc']), args.max_doc_words)
            prompts.append(build_prompt(question, doc_text))
            scores.append(float(item.get('score', 0.0)))
        (prediction, details) = generate_replug(model=model, tokenizer=tokenizer, prompts=prompts, scores=scores, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline, score_temperature=args.replug_score_temperature, fusion=args.replug_fusion, weighting=args.replug_weighting)
        return (prediction, 'replug', 0.0, details)
    if method == 'cad_fixed':
        prediction = generate_cad(model=model, tokenizer=tokenizer, doc_prompt=doc_prompt, no_doc_prompt=no_prompt, alpha=args.alpha, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
        return (prediction, 'cad', float(args.alpha), {})
    if method == 'layer_delta_cad':
        prediction = generate_layer_delta_cad(model=model, tokenizer=tokenizer, doc_prompt=doc_prompt, no_doc_prompt=no_prompt, alpha=args.alpha, contrast_layer=args.contrast_layer, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
        details = {'implementation': 'final_doc_plus_step10r_layer_delta', 'contrast_layer': int(args.contrast_layer), 'contrast_layer_indexing': 'Step10 hidden_states index; 0=embedding, -1/final is final hidden state', 'formula': 'logits_doc_final + alpha * (logits_doc_layer - logits_no_doc_layer)'}
        return (prediction, 'layer_delta_cad', float(args.alpha), details)
    if method == 'acd':
        prediction = generate_acd(model=model, tokenizer=tokenizer, doc_prompt=doc_prompt, no_doc_prompt=no_prompt, alpha=args.alpha, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
        return (prediction, 'acd', float(args.alpha), {'acd_alpha_scale': float(args.alpha)})
    if method == 'context_ucd':
        prediction = generate_context_ucd(model=model, tokenizer=tokenizer, doc_prompt=doc_prompt, no_doc_prompt=no_prompt, beta=args.context_ucd_beta, relative_top=args.context_ucd_relative_top, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
        details = {'implementation': 'context_ucd_energy_weighted_contrast', 'contrast_object': 'same_model_doc_prompt_vs_no_doc_prompt', 'base_prompt': 'selected_topk_doc_prompt', 'amateur_prompt': 'no_doc_prompt', 'energy_temperature': 1.0, 'context_ucd_beta_decay': float(args.context_ucd_beta), 'context_ucd_relative_top': float(args.context_ucd_relative_top), 'sampling': 'disabled_greedy_only'}
        return (prediction, 'context_ucd', 0.0, details)
    if method == 'dola':
        resolved_dola_layers = parse_dola_layers_config(args.dola_layers, args.dola_premature_layer, args.dola_candidate_layers)
        (prediction, details) = generate_dola(model=model, tokenizer=tokenizer, prompt=doc_prompt, dola_layers=resolved_dola_layers, repetition_penalty=args.dola_repetition_penalty, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
        return (prediction, 'dola', 0.0, details)
    raise ValueError(f'Bad decoding method: {method}')

def main() -> None:
    args = parse_args()
    if args.temperature is not None and args.temperature > 0:
        raise ValueError('Use greedy decoding for this controlled experiment: --temperature 0.0')
    if args.max_input_length is not None and args.max_input_length <= 0:
        args.max_input_length = None
    if args.context_ucd_relative_top != 0.0:
        raise ValueError('Context-UCD relative_top is disabled for the main experiment. Use 0.0.')
    if args.dola_premature_layer is not None and args.dola_candidate_layers is not None:
        raise ValueError('Use either --dola-premature-layer or --dola-candidate-layers, not both.')
    (meta, rows) = load_rows(args.input, args.limit)
    dataset = args.dataset or meta.get('dataset') or (rows[0].get('source_dataset') if rows else 'unknown')
    gate_name = sanitize_name(args.selector_name or str(meta.get('selector') or args.rule))
    decoding_method = normalize_decoding_method(args.decoding_method)
    out_path = output_path(args, str(dataset), gate_name, decoding_method)
    if out_path.exists() and (not args.overwrite):
        print(f'[skip completed] {out_path}')
        return
    print(f'[generate] model={args.model} dataset={dataset} gate={gate_name} rule={args.rule} decoding={decoding_method} n={len(rows)}')
    (model, tokenizer, _) = load_base_model_and_tokenizer(args.model)
    outputs = []
    selected_counts = []
    start = time.time()
    for (idx, row) in enumerate(tqdm(rows, desc='top10 gated generation')):
        scored_docs = docs_for_row(row)
        selected = select_docs(scored_docs, args)
        selected_counts.append(len(selected))
        context = build_multi_doc_context(selected, args.max_doc_words)
        has_context = bool(context)
        doc_prompt = build_prompt(row['question'], context)
        no_prompt = build_prompt(row['question'], None)
        (prediction, actual_mode, alpha_used, generation_details) = generate_with_backend(model=model, tokenizer=tokenizer, method=decoding_method, doc_prompt=doc_prompt, no_prompt=no_prompt, has_context=has_context, selected=selected, question=row['question'], args=args)
        selected_scores = [float(x.get('score', 0.0)) for x in selected]
        available_scores = [float(x.get('score', 0.0)) for x in scored_docs]
        p_relevant = float(max(selected_scores, default=0.0))
        trusted = bool(selected)
        outputs.append({'idx': row.get('idx', idx), 'source_dataset': row.get('source_dataset', dataset), 'question': row['question'], 'answers': row.get('answers', []), 'retrieved_doc': selected[0]['doc'] if selected else None, 'available_docs': [x.get('doc') for x in scored_docs], 'selected_docs': [x['doc'] for x in selected], 'selection': {'gate': gate_name, 'rule': args.rule, 'threshold': float(args.threshold), 'select_top_k': int(args.select_top_k), 'num_available_docs': len(scored_docs), 'num_selected_docs': len(selected), 'selected_ranks': [int(x.get('rank', 0)) for x in selected], 'selected_scores': selected_scores, 'max_available_score': float(max(available_scores, default=0.0))}, 'csv': {'p_relevant': p_relevant, 'pred_label': int(trusted), 'pred_name': 'relevant' if trusted else 'distracting', 'margin': None, 'logits': None, 'trusted_by_tau': trusted, 'tau': float(args.threshold), 'gate_source': gate_name}, 'generation': {'method': f'{gate_name}_gate_{decoding_method}', 'gate': gate_name, 'decoding_method': decoding_method, 'actual_mode': actual_mode, 'alpha_requested': float(args.alpha), 'alpha_used': float(alpha_used), 'max_new_tokens': int(args.max_new_tokens), 'temperature': float(args.temperature), 'top_p': float(args.top_p), 'prediction': prediction, 'details': {**generation_details, 'prompt': 'selected_multi_document_rag' if has_context else 'no_doc', 'max_doc_words': int(args.max_doc_words)}}})
    elapsed = time.time() - start
    avg_ctx = sum(selected_counts) / max(1, len(selected_counts))
    trusted_count = int(sum((1 for x in selected_counts if x > 0)))
    method_name = f'{gate_name}_gate_{decoding_method}'
    payload = {'meta': {'model': args.model, 'dataset': dataset, 'method': method_name, 'selector': gate_name, 'gate_method': gate_name, 'decoding_method': decoding_method, 'selection_rule': args.rule, 'threshold': float(args.threshold), 'tau': float(args.threshold), 'select_top_k': int(args.select_top_k), 'alpha': float(args.alpha), 'temperature': float(args.temperature), 'top_p': float(args.top_p), 'max_new_tokens': int(args.max_new_tokens), 'max_input_length': args.max_input_length, 'max_context_docs': int(args.max_context_docs), 'max_doc_words': int(args.max_doc_words), 'input_file': str(args.input), 'num_samples': len(outputs), 'trusted_docs': trusted_count, 'trusted_rate': trusted_count / max(1, len(outputs)), 'avg_selected_docs': avg_ctx, 'num_no_context': int(sum((1 for x in selected_counts if x == 0))), 'elapsed_sec': elapsed, 'context_ucd_beta': float(args.context_ucd_beta) if decoding_method == 'context_ucd' else None, 'context_ucd_relative_top': float(args.context_ucd_relative_top) if decoding_method == 'context_ucd' else None, 'dola_implementation': 'huggingface_generate' if decoding_method == 'dola' else None, 'dola_layers': parse_dola_layers_config(args.dola_layers, args.dola_premature_layer, args.dola_candidate_layers) if decoding_method == 'dola' else None, 'dola_repetition_penalty': float(args.dola_repetition_penalty) if decoding_method == 'dola' else None, 'contrast_layer': int(args.contrast_layer) if decoding_method == 'layer_delta_cad' else None, 'replug_fusion': args.replug_fusion if decoding_method == 'replug' else None, 'replug_weighting': args.replug_weighting if decoding_method == 'replug' else None, 'replug_score_temperature': float(args.replug_score_temperature) if decoding_method == 'replug' else None, 'top10_selection_experiment': True}, 'outputs': outputs}
    dump_json(out_path, payload)
    print(f'[saved] {out_path}')
    print(f"trusted={trusted_count}/{len(outputs)}; avg #ctx={avg_ctx:.3f}; no-context={payload['meta']['num_no_context']}/{len(outputs)}")
if __name__ == '__main__':
    main()
