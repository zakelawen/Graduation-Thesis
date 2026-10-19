from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'final'
RESULTS_DIR = PROJECT_ROOT / 'results'
CSV_DIR = Path(os.environ.get('CSV_RETRIEVED_DIR', RESULTS_DIR / 'csv_retrieved'))
PIPELINE_DIR = RESULTS_DIR / 'pipeline'
sys.path.insert(0, str(PROJECT_ROOT))
from csv_module.llm_layers import add_tsv_layers
from csv_module.train_utils import collate_fn, get_last_non_padded_token_rep
LOCAL_GEMMA2B_PATH = '/home/qluai/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf'
LOCAL_GEMMA9B_PATH = '/root/autodl-tmp/models--google--gemma-2-9b/snapshots/33c193028431c2fde6c6e51f29e6f17b60cbfac6'
LOCAL_QWEN3_4B_PATH = '/home/qluai/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539'
DOLA_CUSTOM_GENERATE = os.environ.get('DOLA_CUSTOM_GENERATE', 'transformers-community/dola')

def resolve_local_custom_generate_file(path_like: str) -> Path | None:
    if not path_like:
        return None
    p = Path(path_like).expanduser()
    if p.is_file() and p.name == 'generate.py':
        return p.resolve()
    candidate = p / 'custom_generate' / 'generate.py'
    if candidate.exists():
        return candidate.resolve()
    return None

def load_local_custom_generate_function(path_like: str):
    generate_py = resolve_local_custom_generate_file(path_like)
    if generate_py is None:
        return None
    module_name = 'local_transformers_community_dola_generate'
    spec = importlib.util.spec_from_file_location(module_name, str(generate_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import local DoLA custom generate file: {generate_py}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'generate'):
        raise RuntimeError(f'Local DoLA custom generate file has no generate() function: {generate_py}')
    return module.generate
MODEL_REGISTRY = {'gemma2b': {'hf_name': os.environ.get('GEMMA_MODEL_PATH', LOCAL_GEMMA2B_PATH), 'csv_run_name': 'gemma2b_retrieved'}, 'gemma9b': {'hf_name': os.environ.get('GEMMA9B_MODEL_PATH', LOCAL_GEMMA9B_PATH), 'csv_run_name': 'gemma9b_retrieved'}, 'qwen3_4b': {'hf_name': os.environ.get('QWEN3_4B_MODEL_PATH', LOCAL_QWEN3_4B_PATH), 'csv_run_name': 'qwen3_4b_retrieved'}}
METHODS = ['no_doc', 'vanilla_rag', 'cad_fixed', 'acd', 'context_ucd', 'dola', 'csv_filter', 'csv_gated_vanilla_rag', 'csv_gated_cad', 'csv_gated_acd', 'csv_gated_context_ucd', 'csv_gated_dola', 'csv_adaptive_cad', 'csv_gated_layer_delta_cad']
DATASETS = ['nq', 'triviaqa']
BASELINE_METHODS = {'no_doc', 'vanilla_rag', 'cad_fixed', 'acd', 'context_ucd', 'dola'}
EXTERNAL_DECODING_METHODS = {'cad_fixed', 'acd', 'context_ucd', 'dola'}
CSV_METHODS = {'csv_filter', 'csv_gated_vanilla_rag', 'csv_gated_cad', 'csv_gated_acd', 'csv_gated_context_ucd', 'csv_gated_dola', 'csv_adaptive_cad', 'csv_gated_layer_delta_cad'}

@dataclass
class CSVCheckpointInfo:
    model_name: str
    csv_run_name: str
    best_key: str
    ckpt_path: Path
    sweep_path: Path
    sweep_entry: dict[str, Any]
    ckpt: dict[str, Any]

@dataclass
class CSVScoreOutput:
    scores: np.ndarray
    preds: np.ndarray
    margins: np.ndarray
    raw_logits: np.ndarray

def load_json(path: Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def extract_doc_text(doc: Any) -> str:
    if isinstance(doc, dict):
        title = str(doc.get('title', '')).strip()
        text = str(doc.get('text', '')).strip()
        if title:
            return f'Title: {title}\nText: {text}'
        return text
    return str(doc).strip()

def build_prompt(question: str, doc_text: str | None) -> str:
    if doc_text is None:
        return f'Question: {question}\nAnswer:'
    return f'Document: {doc_text}\n\nQuestion: {question}\nAnswer:'

def setup_tokenizer(hf_name: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'  Set tokenizer.pad_token = eos_token (id={tokenizer.pad_token_id})')
    else:
        print(f'  tokenizer.pad_token already set: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})')
    return tokenizer

def get_layer_rep(output, attention_mask, cls_layer: int):
    hidden_states = output.hidden_states
    if cls_layer == -1:
        hs = hidden_states[-1]
    else:
        hs = hidden_states[cls_layer + 1]
    return get_last_non_padded_token_rep(hs, attention_mask.to(hs.device))

def current_device(model) -> torch.device:
    return next(model.parameters()).device

def maybe_truncate_left(input_ids: torch.Tensor, max_input_length: int | None) -> torch.Tensor:
    if max_input_length is None:
        return input_ids
    if input_ids.size(1) <= max_input_length:
        return input_ids
    return input_ids[:, -max_input_length:]

def clean_prediction(text: str) -> str:
    text = text.strip()
    if '\n' in text:
        text = text.split('\n', 1)[0].strip()
    for marker in ['Question:', 'Document:', 'Answer:']:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text.strip()

def format_float_for_name(x: float) -> str:
    return str(x).replace('.', 'p').replace('-', 'm')

def resolve_best_csv_checkpoint(model_name: str) -> CSVCheckpointInfo:
    csv_run_name = MODEL_REGISTRY[model_name]['csv_run_name']
    sweep_path = CSV_DIR / f'{csv_run_name}_csv_sweep.json'
    if not sweep_path.exists():
        raise FileNotFoundError(f'Retrieved-doc CSV sweep file not found: {sweep_path}.\nRun Step9R retrieved sweep first, or use a baseline method (no_doc / vanilla_rag / cad_fixed) that does not require CSV.')
    sweep = load_json(sweep_path)
    if not isinstance(sweep, dict) or not sweep:
        raise ValueError(f'Sweep file is empty or malformed: {sweep_path}')
    best_key = max(sweep, key=lambda k: sweep[k]['best_auroc'])
    entry = sweep[best_key]
    raw_ckpt_path = Path(entry['checkpoint'])
    if raw_ckpt_path.exists():
        ckpt_path = raw_ckpt_path
    else:
        fallback = CSV_DIR / raw_ckpt_path.name
        if fallback.exists():
            ckpt_path = fallback
            print('  [path fallback] checkpoint path in sweep does not exist:')
            print(f'    sweep path: {raw_ckpt_path}')
            print(f'    using:      {ckpt_path}')
        else:
            raise FileNotFoundError(f'Checkpoint from retrieved-doc sweep does not exist, and fallback by filename also failed.\n  sweep path: {raw_ckpt_path}\n  fallback:   {fallback}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print('\n=== Best retrieved-doc CSV checkpoint ===')
    print(f'  base model:   {model_name}')
    print(f'  csv run name: {csv_run_name}')
    print(f'  sweep:        {sweep_path}')
    print(f'  best config:  {best_key}')
    print(f"  best AUROC:   {entry['best_auroc']:.6f}")
    print(f"  best Acc:     {entry.get('best_accuracy', float('nan')):.6f}")
    print(f'  checkpoint:   {ckpt_path}')
    return CSVCheckpointInfo(model_name=model_name, csv_run_name=csv_run_name, best_key=best_key, ckpt_path=ckpt_path, sweep_path=sweep_path, sweep_entry=entry, ckpt=ckpt)

def check_checkpoint_metadata(model_name: str, ckpt_info: CSVCheckpointInfo, hidden_size: int):
    ckpt = ckpt_info.ckpt
    required = ['tsv', 'centroids', 'str_layer', 'cls_layer', 'lam', 'cos_temp', 'best_auroc', 'best_accuracy', 'model_name', 'hf_name']
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(f'Checkpoint missing keys: {missing}')
    allowed_ckpt_names = {model_name, ckpt_info.csv_run_name}
    if ckpt['model_name'] not in allowed_ckpt_names:
        raise ValueError(f"Checkpoint model_name={ckpt['model_name']!r} does not match requested model={model_name!r} or retrieved run name {ckpt_info.csv_run_name!r}.")
    tsv = ckpt['tsv']
    centroids = ckpt['centroids']
    if tuple(tsv.shape) != (1, 1, hidden_size):
        raise ValueError(f'Bad TSV shape: {tuple(tsv.shape)}; expected (1, 1, {hidden_size})')
    if tuple(centroids.shape) != (2, hidden_size):
        raise ValueError(f'Bad centroids shape: {tuple(centroids.shape)}; expected (2, {hidden_size})')
    print('\n=== Checkpoint metadata check passed ===')
    print(f"  ckpt model_name: {ckpt['model_name']}")
    print(f'  base model key:  {model_name}')
    print(f'  csv run name:    {ckpt_info.csv_run_name}')
    print(f"  ckpt hf_name:    {ckpt['hf_name']}")
    print(f"  str_layer:       {ckpt['str_layer']}")
    print(f"  cls_layer:       {ckpt['cls_layer']}")
    print(f"  lam:             {ckpt['lam']}")
    print(f"  cos_temp:        {ckpt['cos_temp']}")
    print(f"  best_auroc:      {ckpt['best_auroc']:.6f}")
    print(f"  best_accuracy:   {ckpt['best_accuracy']:.6f}")

def load_base_model_and_tokenizer(model_name: str):
    hf_name = MODEL_REGISTRY[model_name]['hf_name']
    print(f'\n=== Loading model: {model_name} ===')
    print(f'  hf_name: {hf_name}')
    if os.path.isabs(str(hf_name)):
        print(f'  local path exists: {os.path.exists(hf_name)}')
        if not os.path.exists(hf_name):
            raise FileNotFoundError(f'Local model path does not exist for {model_name}: {hf_name}\nSet GEMMA_MODEL_PATH / GEMMA9B_MODEL_PATH / QWEN3_4B_MODEL_PATH to the correct snapshot directory, or edit MODEL_REGISTRY.')
    print('  Loading tokenizer …')
    tokenizer = setup_tokenizer(hf_name)
    print('  Loading model …')
    model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=torch.float16, device_map='auto', low_cpu_mem_usage=True)
    model.eval()
    print(f'  hidden_size={model.config.hidden_size}, num_layers={model.config.num_hidden_layers}')
    print(f'  first parameter device: {current_device(model)}')
    return (model, tokenizer, hf_name)

def inject_csv_into_model(model, ckpt_info: CSVCheckpointInfo):
    ckpt = ckpt_info.ckpt
    tsv_param = nn.Parameter(ckpt['tsv'].float(), requires_grad=False)
    alpha = [float(ckpt['lam'])]
    add_tsv_layers(model, tsv_param, alpha, int(ckpt['str_layer']), ckpt_info.model_name)
    print('\n=== CSV vector injected for scoring ===')
    print(f"  inject layer: {ckpt['str_layer']}")
    print(f"  lambda:       {ckpt['lam']}")

def prepare_eval_prompts(tokenizer):
    eval_path = PROJECT_ROOT / 'data' / 'final_retrieved' / 'eval.json'
    if not eval_path.exists():
        raise FileNotFoundError(f'Cannot find {eval_path}. Run Step4b/Step9R first.')
    data = load_json(eval_path)
    prompts = []
    labels = []
    sources = []
    questions = []
    for (idx, item) in enumerate(data):
        q = item['question']
        src = str(item.get('source_dataset', 'unknown')).lower().strip()
        if 'doc' not in item:
            raise KeyError(f"Retrieved eval item {idx} has no 'doc' field.")
        if 'label' in item:
            label = int(item['label'])
        else:
            label_name = str(item.get('label_name', '')).lower().strip()
            if label_name == 'relevant':
                label = 1
            elif label_name == 'distracting':
                label = 0
            else:
                raise ValueError(f'Retrieved eval item {idx} has invalid label_name={label_name!r}')
        if label not in (0, 1):
            raise ValueError(f'Retrieved eval item {idx} has invalid label={label!r}')
        doc_text = extract_doc_text(item['doc'])
        prompt = build_prompt(q, doc_text)
        prompts.append(tokenizer(prompt, return_tensors='pt').input_ids)
        labels.append(label)
        sources.append(src)
        questions.append(q)
    print(f'  Loaded retrieved eval: {len(data)} samples → {len(prompts)} prompts')
    return (prompts, labels, sources, questions)

def prepare_doc_scoring_prompts(tokenizer, samples: list[dict[str, Any]]):
    prompts = []
    for item in samples:
        q = item['question']
        doc_text = extract_doc_text(item['retrieved_doc'])
        prompt = build_prompt(q, doc_text)
        prompts.append(tokenizer(prompt, return_tensors='pt').input_ids)
    return prompts

@torch.no_grad()
def score_prompts_with_csv(model, prompts: list[torch.Tensor], batch_size: int, pad_id: int, cls_layer: int, cos_temp: float, centroids: torch.Tensor) -> CSVScoreOutput:
    device = current_device(model)
    model.eval()
    all_scores = []
    all_preds = []
    all_margins = []
    all_logits = []
    centroids = centroids.to(device).float()
    for start in tqdm(range(0, len(prompts), batch_size), desc='CSV scoring'):
        batch_p_list = prompts[start:start + batch_size]
        fake_labels = [0] * len(batch_p_list)
        (batch_p, _, attention_mask) = collate_fn(batch_p_list, fake_labels, pad_id=pad_id)
        batch_p = batch_p.to(device)
        attention_mask = attention_mask.to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = model.model(input_ids=batch_p, attention_mask=attention_mask, output_hidden_states=True)
            last_token_rep = get_layer_rep(output, attention_mask, cls_layer)
        last_token_rep = F.normalize(last_token_rep.float(), p=2, dim=-1)
        c = F.normalize(centroids.to(last_token_rep.device), p=2, dim=-1)
        sims = torch.matmul(last_token_rep, c.T)
        logits = sims / cos_temp
        probs = torch.softmax(logits, dim=-1)
        scores = probs[:, 1]
        preds = torch.argmax(probs, dim=-1)
        margins = torch.abs(logits[:, 1] - logits[:, 0])
        all_scores.append(scores.cpu())
        all_preds.append(preds.cpu())
        all_margins.append(margins.cpu())
        all_logits.append(logits.cpu())
    return CSVScoreOutput(scores=torch.cat(all_scores).numpy(), preds=torch.cat(all_preds).numpy(), margins=torch.cat(all_margins).numpy(), raw_logits=torch.cat(all_logits).numpy())

def validate_csv_full_eval(model, tokenizer, ckpt_info: CSVCheckpointInfo, batch_size: int, tolerance: float) -> dict[str, Any]:
    print('\n=== Full eval validation for CSV scorer ===')
    ckpt = ckpt_info.ckpt
    (prompts, labels, sources, _) = prepare_eval_prompts(tokenizer)
    scored = score_prompts_with_csv(model=model, prompts=prompts, batch_size=batch_size, pad_id=tokenizer.pad_token_id, cls_layer=int(ckpt['cls_layer']), cos_temp=float(ckpt['cos_temp']), centroids=ckpt['centroids'])
    labels_arr = np.array(labels, dtype=np.int64)
    sources_arr = np.array(sources)
    metrics = {}
    for subset in sorted(set(sources)) + ['combined']:
        if subset == 'combined':
            mask = np.ones(len(labels_arr), dtype=bool)
        else:
            mask = sources_arr == subset
        if not mask.any():
            continue
        y = labels_arr[mask]
        s = scored.scores[mask]
        p = scored.preds[mask]
        m = scored.margins[mask]
        metrics[subset] = {'auroc': float(roc_auc_score(y, s)), 'accuracy': float(accuracy_score(y, p)), 'avg_margin': float(m.mean()), 'n_prompts': int(mask.sum())}
    combined = metrics['combined']
    saved_best_auroc = float(ckpt_info.sweep_entry.get('best_auroc', ckpt['best_auroc']))
    diff = abs(combined['auroc'] - saved_best_auroc)
    print('  Validation metrics:')
    for (subset, vals) in metrics.items():
        print(f"    {subset:8s}: AUROC={vals['auroc']:.6f}  Acc={vals['accuracy']:.6f}  Margin={vals['avg_margin']:.4f}  N={vals['n_prompts']}")
    print(f'  Saved best AUROC: {saved_best_auroc:.6f}')
    print(f'  AUROC diff:       {diff:.8f}')
    validation = {'model_name': ckpt_info.model_name, 'best_key': ckpt_info.best_key, 'sweep_path': str(ckpt_info.sweep_path), 'checkpoint': str(ckpt_info.ckpt_path), 'str_layer': int(ckpt['str_layer']), 'cls_layer': int(ckpt['cls_layer']), 'lam': float(ckpt['lam']), 'cos_temp': float(ckpt['cos_temp']), 'saved_best_auroc': saved_best_auroc, 'saved_best_accuracy': float(ckpt_info.sweep_entry.get('best_accuracy', ckpt['best_accuracy'])), 'computed': metrics, 'auroc_diff': float(diff), 'tolerance': float(tolerance), 'passed': bool(diff <= tolerance)}
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PIPELINE_DIR / f'{ckpt_info.csv_run_name}_csv_eval_validation.json'
    save_json(validation, out_path)
    print(f'  Saved validation → {out_path}')
    if diff > tolerance:
        raise RuntimeError(f'CSV validation failed: computed AUROC differs from saved best by {diff:.6f}, larger than tolerance={tolerance}.')
    print('  ✓ CSV validation passed')
    return validation

def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    (sorted_logits, sorted_indices) = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, -float('inf'))
    return logits

def select_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature is not None and temperature > 0:
        raise ValueError('This Step11 script is configured for greedy decoding only. Use --temperature 0.0. Sampling is intentionally disabled.')
    return torch.argmax(logits, dim=-1, keepdim=True)

def _tokenize_prompt_for_generation(tokenizer, prompt: str, device: torch.device, max_input_length: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors='pt')
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.get('attention_mask', torch.ones_like(encoded.input_ids)).to(device)
    if max_input_length is not None and input_ids.size(1) > max_input_length:
        input_ids = input_ids[:, -max_input_length:]
        attention_mask = attention_mask[:, -max_input_length:]
    return (input_ids, attention_mask)

def _append_attention_token(attention_mask: torch.Tensor) -> torch.Tensor:
    return torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)

@torch.no_grad()
def _init_cached_generation_state(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    return (out.logits[:, -1, :], out.past_key_values, attention_mask)

@torch.no_grad()
def _advance_cached_generation_state(model, next_id: torch.Tensor, past_key_values, attention_mask: torch.Tensor):
    attention_mask = _append_attention_token(attention_mask)
    out = model(input_ids=next_id, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
    return (out.logits[:, -1, :], out.past_key_values, attention_mask)

def _project_last_token_layer_logits(model, hidden_states, layer_idx: int) -> torch.Tensor:
    num_hidden_states = len(hidden_states)
    if layer_idx < 0:
        layer_idx = num_hidden_states + layer_idx
    if layer_idx < 0 or layer_idx >= num_hidden_states:
        raise ValueError(f'contrast_layer={layer_idx} is out of range for {num_hidden_states} hidden states')
    hs = hidden_states[layer_idx][:, -1:, :]
    layer_logits = project_hidden_to_logits_for_dola(model=model, hs=hs, layer_idx=layer_idx, num_hidden_states=num_hidden_states)
    return layer_logits[:, -1, :]

@torch.no_grad()
def _init_layer_delta_generation_state(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, contrast_layer: int):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, output_hidden_states=True)
    final_logits = out.logits[:, -1, :]
    layer_logits = _project_last_token_layer_logits(model=model, hidden_states=out.hidden_states, layer_idx=contrast_layer)
    return (final_logits, layer_logits, out.past_key_values, attention_mask)

@torch.no_grad()
def _advance_layer_delta_generation_state(model, next_id: torch.Tensor, past_key_values, attention_mask: torch.Tensor, contrast_layer: int):
    attention_mask = _append_attention_token(attention_mask)
    out = model(input_ids=next_id, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
    final_logits = out.logits[:, -1, :]
    layer_logits = _project_last_token_layer_logits(model=model, hidden_states=out.hidden_states, layer_idx=contrast_layer)
    return (final_logits, layer_logits, out.past_key_values, attention_mask)

@torch.no_grad()
def generate_standard(model, tokenizer, prompt: str, max_new_tokens: int, max_input_length: int | None, temperature: float, top_p: float, stop_on_newline: bool) -> str:
    device = current_device(model)
    (ids, attention_mask) = _tokenize_prompt_for_generation(tokenizer, prompt, device, max_input_length)
    (logits, past, attention_mask) = _init_cached_generation_state(model, ids, attention_mask)
    generated = []
    eos_id = tokenizer.eos_token_id
    for step_idx in range(max_new_tokens):
        next_id = select_next_token(logits, temperature, top_p)
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
        (logits, past, attention_mask) = _advance_cached_generation_state(model, next_id, past, attention_mask)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return clean_prediction(text)

@torch.no_grad()
def generate_cad(model, tokenizer, doc_prompt: str, no_doc_prompt: str, alpha: float, max_new_tokens: int, max_input_length: int | None, temperature: float, top_p: float, stop_on_newline: bool) -> str:
    device = current_device(model)
    (doc_ids, doc_mask) = _tokenize_prompt_for_generation(tokenizer, doc_prompt, device, max_input_length)
    (no_ids, no_mask) = _tokenize_prompt_for_generation(tokenizer, no_doc_prompt, device, max_input_length)
    (logits_doc, past_doc, doc_mask) = _init_cached_generation_state(model, doc_ids, doc_mask)
    (logits_no, past_no, no_mask) = _init_cached_generation_state(model, no_ids, no_mask)
    generated = []
    eos_id = tokenizer.eos_token_id
    for step_idx in range(max_new_tokens):
        adjusted_logits = (1.0 + alpha) * logits_doc - alpha * logits_no
        next_id = select_next_token(adjusted_logits, temperature, top_p)
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
        (logits_doc, past_doc, doc_mask) = _advance_cached_generation_state(model, next_id, past_doc, doc_mask)
        (logits_no, past_no, no_mask) = _advance_cached_generation_state(model, next_id, past_no, no_mask)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return clean_prediction(text)

@torch.no_grad()
def generate_layer_delta_cad(model, tokenizer, doc_prompt: str, no_doc_prompt: str, alpha: float, contrast_layer: int, max_new_tokens: int, max_input_length: int | None, temperature: float, top_p: float, stop_on_newline: bool) -> str:
    device = current_device(model)
    (doc_ids, doc_mask) = _tokenize_prompt_for_generation(tokenizer, doc_prompt, device, max_input_length)
    (no_ids, no_mask) = _tokenize_prompt_for_generation(tokenizer, no_doc_prompt, device, max_input_length)
    (logits_doc_final, logits_doc_layer, past_doc, doc_mask) = _init_layer_delta_generation_state(model, doc_ids, doc_mask, contrast_layer)
    (_, logits_no_layer, past_no, no_mask) = _init_layer_delta_generation_state(model, no_ids, no_mask, contrast_layer)
    generated = []
    eos_id = tokenizer.eos_token_id
    for step_idx in range(max_new_tokens):
        adjusted_logits = logits_doc_final + float(alpha) * (logits_doc_layer - logits_no_layer)
        next_id = select_next_token(adjusted_logits, temperature, top_p)
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
        (logits_doc_final, logits_doc_layer, past_doc, doc_mask) = _advance_layer_delta_generation_state(model, next_id, past_doc, doc_mask, contrast_layer)
        (_, logits_no_layer, past_no, no_mask) = _advance_layer_delta_generation_state(model, next_id, past_no, no_mask, contrast_layer)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return clean_prediction(text)

def entropy_from_logits(logits: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + eps)
    return -(probs * log_probs).sum(dim=-1)

def get_final_norm(model):
    inner = model.model
    if hasattr(inner, 'norm'):
        return inner.norm
    raise RuntimeError(f'Cannot locate final norm on {type(inner).__name__}; inspect model.model to add architecture-specific support.')

def apply_final_logit_softcapping(logits: torch.Tensor, softcap: float | None):
    if softcap is None:
        return logits
    return softcap * torch.tanh(logits / softcap)

def project_hidden_to_logits_for_dola(model, hs: torch.Tensor, layer_idx: int, num_hidden_states: int):
    if layer_idx == num_hidden_states - 1:
        hs_for_lm = hs
    else:
        final_norm = get_final_norm(model)
        hs_for_lm = final_norm(hs)
    logits = model.lm_head(hs_for_lm)
    softcap = getattr(model.config, 'final_logit_softcapping', None)
    return apply_final_logit_softcapping(logits, softcap)

def transformer_layer_to_hidden_state_index(layer: int, num_hidden_states: int) -> int:
    if layer == -1:
        return num_hidden_states - 1
    idx = layer + 1
    if idx < 0 or idx >= num_hidden_states:
        raise ValueError(f'Bad layer={layer}; model returned {num_hidden_states} hidden states so valid transformer layers are 0..{num_hidden_states - 2}, or -1.')
    return idx

def parse_int_list(text: str | None) -> list[int] | None:
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    return [int(x.strip()) for x in text.split(',') if x.strip()]

def parse_dola_layers_config(dola_layers: str | None, legacy_premature_layer: int | None, legacy_candidate_layers: str | None) -> str | list[int]:
    if dola_layers is not None and str(dola_layers).strip():
        text = str(dola_layers).strip().lower()
        if text in {'low', 'high'}:
            return text
        return parse_int_list(dola_layers) or 'high'
    cand = parse_int_list(legacy_candidate_layers)
    if cand is not None:
        return cand
    if legacy_premature_layer is not None:
        return [int(legacy_premature_layer)]
    return 'high'

def default_dola_candidate_layers(num_transformer_layers: int) -> list[int]:
    upper = max(1, num_transformer_layers // 2)
    cand = list(range(0, upper, 2))
    return cand or [0]

def relative_top_filter_original(scores: torch.Tensor, relative_top: float, filter_value: float=-float('inf'), min_tokens_to_keep: int=1) -> torch.Tensor:
    scores_normalized = scores.float().log_softmax(dim=-1)
    if relative_top is None or relative_top <= 0.0:
        return scores_normalized
    (sorted_logits, _) = torch.sort(scores_normalized, descending=True, dim=-1)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + math.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh).unsqueeze(-1)
    return scores_normalized.masked_fill(scores_normalized < probs_thresh, filter_value)

def relative_top_filter_probs_ucd(scores: torch.Tensor, relative_top: float, filter_value: float=0.0, min_tokens_to_keep: int=1) -> torch.Tensor:
    probs = scores.float().softmax(dim=-1)
    if relative_top is None or relative_top <= 0.0:
        return probs
    (sorted_probs, _) = torch.sort(probs, descending=True, dim=-1)
    min_thresh = sorted_probs[..., min_tokens_to_keep - 1]
    probs_max = torch.max(probs, dim=-1).values
    probs_thresh = probs_max + math.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh).unsqueeze(-1)
    return probs.masked_fill(probs < probs_thresh, filter_value)

def mask_to_anchor_relative_top(adjusted_logits: torch.Tensor, anchor_logits: torch.Tensor, relative_top: float, filter_value: float=-float('inf'), min_tokens_to_keep: int=1) -> torch.Tensor:
    if relative_top is None or relative_top <= 0.0:
        return adjusted_logits
    anchor_filtered = relative_top_filter_original(anchor_logits, relative_top, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)
    return adjusted_logits.masked_fill(anchor_filtered <= filter_value / 2, filter_value)

@torch.no_grad()
def generate_acd(model, tokenizer, doc_prompt: str, no_doc_prompt: str, alpha: float, max_new_tokens: int, max_input_length: int | None, temperature: float, top_p: float, stop_on_newline: bool) -> str:
    device = current_device(model)
    (doc_ids, doc_mask) = _tokenize_prompt_for_generation(tokenizer, doc_prompt, device, max_input_length)
    (no_ids, no_mask) = _tokenize_prompt_for_generation(tokenizer, no_doc_prompt, device, max_input_length)
    (logits_doc, past_doc, doc_mask) = _init_cached_generation_state(model, doc_ids, doc_mask)
    (logits_no, past_no, no_mask) = _init_cached_generation_state(model, no_ids, no_mask)
    generated = []
    eos_id = tokenizer.eos_token_id
    for step_idx in range(max_new_tokens):
        ent_no = entropy_from_logits(logits_no)
        ent_doc = entropy_from_logits(logits_doc)
        d_alpha = (ent_no / (ent_doc + ent_no + 1e-12)).unsqueeze(-1)
        adjusted_logits = logits_no + float(alpha) * d_alpha * (logits_doc - logits_no)
        next_id = select_next_token(adjusted_logits, temperature, top_p)
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
        (logits_doc, past_doc, doc_mask) = _advance_cached_generation_state(model, next_id, past_doc, doc_mask)
        (logits_no, past_no, no_mask) = _advance_cached_generation_state(model, next_id, past_no, no_mask)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return clean_prediction(text)

def calculate_ucd_energy(logits: torch.Tensor, logit_prev: torch.Tensor, energy_temperature: float=1.0) -> torch.Tensor:
    if energy_temperature <= 0:
        raise ValueError('energy_temperature must be > 0')
    adjusted = logits.float() + logit_prev.unsqueeze(-1).float()
    return energy_temperature * torch.logsumexp(adjusted / energy_temperature, dim=-1)

@torch.no_grad()
def generate_context_ucd(model, tokenizer, doc_prompt: str, no_doc_prompt: str, beta: float, relative_top: float, max_new_tokens: int, max_input_length: int | None, temperature: float, top_p: float, stop_on_newline: bool) -> str:
    if temperature is not None and temperature > 0:
        raise ValueError('Context-UCD-Energy is greedy-only. Use --temperature 0.0.')
    if relative_top is not None and float(relative_top) != 0.0:
        raise ValueError('Context-UCD-Energy relative_top is disabled in the main implementation. Use --context_ucd_relative_top 0.0.')
    device = current_device(model)
    (doc_ids, doc_mask) = _tokenize_prompt_for_generation(tokenizer, doc_prompt, device, max_input_length)
    (no_ids, no_mask) = _tokenize_prompt_for_generation(tokenizer, no_doc_prompt, device, max_input_length)
    (logits_doc, past_doc, doc_mask) = _init_cached_generation_state(model, doc_ids, doc_mask)
    (logits_no, past_no, no_mask) = _init_cached_generation_state(model, no_ids, no_mask)
    generated = []
    eos_id = tokenizer.eos_token_id
    logit_doc_prev = torch.zeros(logits_doc.size(0), device=device, dtype=torch.float32)
    logit_no_prev = torch.zeros(logits_no.size(0), device=device, dtype=torch.float32)
    energy_temperature = 1.0
    for step_idx in range(max_new_tokens):
        e_doc = calculate_ucd_energy(logits=logits_doc, logit_prev=logit_doc_prev, energy_temperature=energy_temperature)
        e_no = calculate_ucd_energy(logits=logits_no, logit_prev=logit_no_prev, energy_temperature=energy_temperature)
        denom = e_doc + e_no + 1e-12
        w_doc = (e_doc / denom).unsqueeze(-1)
        w_no = (e_no / denom).unsqueeze(-1)
        adjusted_logits = 2.0 * w_doc * logits_doc.float() - w_no * logits_no.float()
        next_id = select_next_token(adjusted_logits, temperature, top_p)
        token_id = int(next_id.item())
        if eos_id is not None and token_id == eos_id:
            break
        chosen_doc_logit = logits_doc.float().gather(1, next_id).squeeze(-1).detach()
        chosen_no_logit = logits_no.float().gather(1, next_id).squeeze(-1).detach()
        logit_doc_prev = float(beta) * logit_doc_prev + chosen_doc_logit
        logit_no_prev = float(beta) * logit_no_prev + chosen_no_logit
        generated.append(token_id)
        if stop_on_newline:
            text = tokenizer.decode(generated, skip_special_tokens=True)
            if '\n' in text:
                break
        if step_idx == max_new_tokens - 1:
            break
        (logits_doc, past_doc, doc_mask) = _advance_cached_generation_state(model, next_id, past_doc, doc_mask)
        (logits_no, past_no, no_mask) = _advance_cached_generation_state(model, next_id, past_no, no_mask)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return clean_prediction(text)

def js_divergence_for_logits(mature_logits: torch.Tensor, premature_logits: torch.Tensor) -> torch.Tensor:
    p_m = torch.softmax(mature_logits.float(), dim=-1)
    p_p = torch.softmax(premature_logits.float(), dim=-1)
    m = 0.5 * (p_m + p_p)
    kl_m = F.kl_div(torch.log(p_m + 1e-12), m, reduction='batchmean')
    kl_p = F.kl_div(torch.log(p_p + 1e-12), m, reduction='batchmean')
    return 0.5 * (kl_m + kl_p)

@torch.no_grad()
def generate_dola(model, tokenizer, prompt: str, dola_layers: str | list[int], repetition_penalty: float, max_new_tokens: int, max_input_length: int | None, temperature: float, top_p: float, stop_on_newline: bool) -> tuple[str, dict[str, Any]]:
    device = current_device(model)
    encoded = tokenizer(prompt, return_tensors='pt')
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.get('attention_mask', torch.ones_like(encoded.input_ids)).to(device)
    if max_input_length is not None and input_ids.size(1) > max_input_length:
        input_ids = input_ids[:, -max_input_length:]
        attention_mask = attention_mask[:, -max_input_length:]
    if temperature is not None and temperature > 0:
        raise ValueError('DoLA is run with greedy decoding only. Use --temperature 0.0.')
    gen_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'max_new_tokens': int(max_new_tokens), 'do_sample': False, 'custom_generate': DOLA_CUSTOM_GENERATE, 'trust_remote_code': True, 'dola_layers': dola_layers, 'repetition_penalty': float(repetition_penalty), 'pad_token_id': tokenizer.pad_token_id, 'eos_token_id': tokenizer.eos_token_id}
    old_model_output_hidden_states = getattr(model.config, 'output_hidden_states', None)
    model.config.output_hidden_states = True
    old_gen_output_hidden_states = None
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        old_gen_output_hidden_states = getattr(model.generation_config, 'output_hidden_states', None)
        try:
            model.generation_config.output_hidden_states = True
        except Exception:
            pass
    try:
        local_generate_fn = load_local_custom_generate_function(DOLA_CUSTOM_GENERATE)
        if local_generate_fn is not None:
            local_kwargs = dict(gen_kwargs)
            local_kwargs.pop('custom_generate', None)
            local_kwargs.pop('trust_remote_code', None)
            output_ids = local_generate_fn(model=model, **local_kwargs)
        else:
            output_ids = model.generate(**gen_kwargs)
    except TypeError as e:
        msg = str(e)
        if 'custom_generate' in msg or 'trust_remote_code' in msg:
            fallback_kwargs = dict(gen_kwargs)
            fallback_kwargs.pop('custom_generate', None)
            fallback_kwargs.pop('trust_remote_code', None)
            output_ids = model.generate(**fallback_kwargs)
        else:
            raise RuntimeError(f"This installed Transformers version/model does not appear to support DoLA generation. For recent Transformers, use custom_generate='transformers-community/dola', trust_remote_code=True, and output_hidden_states=True; for older versions, verify built-in dola_layers support. Original error: {e}") from e
    except ValueError as e:
        msg = str(e)
        if 'trust_remote_code=True' in msg or 'custom_generate' in msg:
            raise RuntimeError(f'DoLA is now loaded from the HuggingFace custom_generate repo `transformers-community/dola`. If using a local copy, set DOLA_CUSTOM_GENERATE to either the repo directory containing custom_generate/generate.py or the generate.py file itself. If using the Hub, connect to HF once or pre-cache transformers-community/dola. Original error: {e}') from e
        if 'dola_layers' in msg or 'not used by the model' in msg:
            raise RuntimeError(f'This installed Transformers version/model rejected `dola_layers`. Upgrade Transformers or verify that this model class supports DoLA/custom_generate. Original error: {e}') from e
        raise
    finally:
        model.config.output_hidden_states = old_model_output_hidden_states
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            try:
                model.generation_config.output_hidden_states = old_gen_output_hidden_states
            except Exception:
                pass
    new_tokens = output_ids[0, input_ids.size(1):]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    details = {'implementation': 'huggingface_generate', 'dola_layers': dola_layers, 'repetition_penalty': float(repetition_penalty), 'stop_on_newline': bool(stop_on_newline)}
    return (clean_prediction(text), details)

def decide_generation_mode(method: str, p_relevant: float, tau: float, alpha: float) -> tuple[str, float, bool]:
    trusted = p_relevant >= tau
    if method == 'no_doc':
        return ('no_doc', 0.0, trusted)
    if method == 'vanilla_rag':
        return ('vanilla_rag', 0.0, trusted)
    if method == 'cad_fixed':
        return ('cad', alpha, trusted)
    if method == 'acd':
        return ('acd', alpha, trusted)
    if method == 'context_ucd':
        return ('context_ucd', 0.0, trusted)
    if method == 'dola':
        return ('dola', 0.0, trusted)
    if method in {'csv_filter', 'csv_gated_vanilla_rag'}:
        return ('vanilla_rag', 0.0, trusted) if trusted else ('no_doc', 0.0, trusted)
    if method == 'csv_gated_cad':
        return ('cad', alpha, trusted) if trusted else ('no_doc', 0.0, trusted)
    if method == 'csv_gated_acd':
        return ('acd', alpha, trusted) if trusted else ('no_doc', 0.0, trusted)
    if method == 'csv_gated_context_ucd':
        return ('context_ucd', 0.0, trusted) if trusted else ('no_doc', 0.0, trusted)
    if method == 'csv_gated_dola':
        return ('dola', 0.0, trusted) if trusted else ('no_doc', 0.0, trusted)
    if method == 'csv_gated_layer_delta_cad':
        return ('layer_delta_cad', alpha, trusted) if trusted else ('no_doc', 0.0, trusted)
    if method == 'csv_adaptive_cad':
        if not trusted:
            return ('no_doc', 0.0, trusted)
        denom = max(1e-08, 1.0 - tau)
        alpha_used = alpha * max(0.0, min(1.0, (p_relevant - tau) / denom))
        return ('cad', float(alpha_used), trusted)
    raise ValueError(f'Unknown method: {method}')

def load_test_samples(dataset: str, limit: int | None=None) -> list[dict[str, Any]]:
    path = DATA_DIR / f'test_retrieval_{dataset}.json'
    if not path.exists():
        raise FileNotFoundError(f'Cannot find {path}. Run Step5 first.')
    samples = load_json(path)
    for s in samples:
        s['source_dataset'] = dataset
    if limit is not None:
        samples = samples[:limit]
    print(f'  Loaded test_retrieval_{dataset}: {len(samples)} samples')
    return samples

def run_generation_for_dataset(model, tokenizer, dataset: str, samples: list[dict[str, Any]], csv_scores: CSVScoreOutput | None, args, ckpt_info: CSVCheckpointInfo | None, validation: dict[str, Any] | None) -> Path:
    print(f'\n=== Running generation: dataset={dataset}, method={args.method} ===')
    needs_csv = args.method in CSV_METHODS
    if needs_csv and csv_scores is None:
        raise ValueError(f'Method {args.method} requires CSV scores, but csv_scores=None')
    outputs = []
    start_time = time.time()
    for (i, item) in enumerate(tqdm(samples, desc=f'generate {dataset}')):
        q = item['question']
        answers = item.get('answers', [])
        doc_text = extract_doc_text(item['retrieved_doc'])
        doc_prompt = build_prompt(q, doc_text)
        no_prompt = build_prompt(q, None)
        if csv_scores is None:
            p_rel = 0.0
            pred = None
            margin = None
            logits = None
            (mode, alpha_used, trusted) = decide_generation_mode(method=args.method, p_relevant=p_rel, tau=args.tau, alpha=args.alpha)
            csv_obj = None
        else:
            p_rel = float(csv_scores.scores[i])
            pred = int(csv_scores.preds[i])
            margin = float(csv_scores.margins[i])
            logits = csv_scores.raw_logits[i].tolist()
            (mode, alpha_used, trusted) = decide_generation_mode(method=args.method, p_relevant=p_rel, tau=args.tau, alpha=args.alpha)
            csv_obj = {'p_relevant': p_rel, 'pred_label': pred, 'pred_name': 'relevant' if pred == 1 else 'distracting', 'margin': margin, 'logits': logits, 'trusted_by_tau': bool(trusted), 'tau': float(args.tau)}
        generation_details = {}
        if mode == 'no_doc':
            prediction = generate_standard(model=model, tokenizer=tokenizer, prompt=no_prompt, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
        elif mode == 'vanilla_rag':
            prediction = generate_standard(model=model, tokenizer=tokenizer, prompt=doc_prompt, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
        elif mode == 'cad':
            prediction = generate_cad(model=model, tokenizer=tokenizer, doc_prompt=doc_prompt, no_doc_prompt=no_prompt, alpha=alpha_used, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
        elif mode == 'layer_delta_cad':
            prediction = generate_layer_delta_cad(model=model, tokenizer=tokenizer, doc_prompt=doc_prompt, no_doc_prompt=no_prompt, alpha=alpha_used, contrast_layer=args.contrast_layer, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
            generation_details = {'implementation': 'final_doc_plus_step10r_layer_delta', 'contrast_layer': int(args.contrast_layer), 'contrast_layer_indexing': 'Step10 hidden_states index; 0=embedding, -1/final is final hidden state', 'formula': 'logits_doc_final + alpha * (logits_doc_layer - logits_no_doc_layer)'}
        elif mode == 'acd':
            prediction = generate_acd(model=model, tokenizer=tokenizer, doc_prompt=doc_prompt, no_doc_prompt=no_prompt, alpha=alpha_used, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
            generation_details = {'acd_alpha_scale': float(alpha_used)}
        elif mode == 'context_ucd':
            prediction = generate_context_ucd(model=model, tokenizer=tokenizer, doc_prompt=doc_prompt, no_doc_prompt=no_prompt, beta=args.context_ucd_beta, relative_top=args.context_ucd_relative_top, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
            generation_details = {'implementation': 'context_ucd_energy_weighted_contrast', 'contrast_object': 'same_model_doc_prompt_vs_no_doc_prompt', 'base_prompt': 'doc_prompt', 'amateur_prompt': 'no_doc_prompt', 'energy_temperature': 1.0, 'context_ucd_beta_decay': float(args.context_ucd_beta), 'context_ucd_relative_top': float(args.context_ucd_relative_top), 'sampling': 'disabled_greedy_only'}
        elif mode == 'dola':
            resolved_dola_layers = parse_dola_layers_config(dola_layers=args.dola_layers, legacy_premature_layer=args.dola_premature_layer, legacy_candidate_layers=args.dola_candidate_layers)
            (prediction, generation_details) = generate_dola(model=model, tokenizer=tokenizer, prompt=doc_prompt, dola_layers=resolved_dola_layers, repetition_penalty=args.dola_repetition_penalty, max_new_tokens=args.max_new_tokens, max_input_length=args.max_input_length, temperature=args.temperature, top_p=args.top_p, stop_on_newline=args.stop_on_newline)
        else:
            raise ValueError(f'Bad mode: {mode}')
        outputs.append({'idx': i, 'source_dataset': dataset, 'question': q, 'answers': answers, 'retrieved_doc': item['retrieved_doc'], 'csv': csv_obj, 'generation': {'method': args.method, 'actual_mode': mode, 'alpha_requested': float(args.alpha), 'alpha_used': float(alpha_used), 'max_new_tokens': int(args.max_new_tokens), 'temperature': float(args.temperature), 'top_p': float(args.top_p), 'prediction': prediction, 'details': generation_details}})
    elapsed = time.time() - start_time
    if csv_scores is None:
        trusted_count = 0
        trusted_rate = None
    else:
        trusted_count = int(sum((1 for o in outputs if o['csv']['trusted_by_tau'])))
        trusted_rate = trusted_count / max(1, len(outputs))
    csv_validation_path = None
    csv_validation_passed = None
    csv_best_key = None
    csv_checkpoint = None
    if ckpt_info is not None:
        csv_validation_path = str(PIPELINE_DIR / f'{ckpt_info.csv_run_name}_csv_eval_validation.json')
        csv_best_key = ckpt_info.best_key
        csv_checkpoint = str(ckpt_info.ckpt_path)
    if validation is not None:
        csv_validation_passed = validation['passed']
    result_obj = {'meta': {'model': args.model, 'dataset': dataset, 'method': args.method, 'alpha': float(args.alpha), 'tau': float(args.tau), 'temperature': float(args.temperature), 'top_p': float(args.top_p), 'max_new_tokens': int(args.max_new_tokens), 'max_input_length': args.max_input_length, 'limit': args.limit, 'num_samples': len(outputs), 'needs_csv': bool(needs_csv), 'csv_source': 'retrieved_doc_step9r' if needs_csv else None, 'contrast_layer': int(args.contrast_layer) if args.method == 'csv_gated_layer_delta_cad' else None, 'contrast_layer_source': 'step10r_retrieved_logit_lens' if args.method == 'csv_gated_layer_delta_cad' else None, 'trusted_docs': trusted_count, 'trusted_rate': trusted_rate, 'elapsed_sec': elapsed, 'csv_best_key': csv_best_key, 'csv_checkpoint': csv_checkpoint, 'csv_validation_path': csv_validation_path, 'csv_validation_passed': csv_validation_passed, 'external_decoding_method': args.method if args.method in EXTERNAL_DECODING_METHODS else None, 'context_ucd_beta': float(args.context_ucd_beta) if args.method == 'context_ucd' else None, 'context_ucd_relative_top': float(args.context_ucd_relative_top) if args.method == 'context_ucd' else None, 'context_ucd_definition': 'same-model UCD-energy contrast: doc_prompt as base vs no_doc_prompt as amateur' if args.method == 'context_ucd' else None, 'dola_implementation': 'huggingface_generate' if args.method == 'dola' else None, 'dola_layers': parse_dola_layers_config(args.dola_layers, args.dola_premature_layer, args.dola_candidate_layers) if args.method == 'dola' else None, 'dola_repetition_penalty': float(args.dola_repetition_penalty) if args.method == 'dola' else None, 'dola_mature_layer_legacy': int(args.dola_mature_layer) if args.method == 'dola' else None, 'dola_premature_layer_legacy': args.dola_premature_layer if args.method == 'dola' else None, 'dola_candidate_layers_legacy': args.dola_candidate_layers if args.method == 'dola' else None, 'dola_relative_top_legacy': float(args.dola_relative_top) if args.method == 'dola' else None}, 'outputs': outputs}
    alpha_name = format_float_for_name(args.alpha)
    tau_name = format_float_for_name(args.tau)
    temp_name = format_float_for_name(args.temperature)
    limit_suffix = f'_limit{args.limit}' if args.limit is not None else ''
    layer_suffix = f'_layer{args.contrast_layer}' if args.method == 'csv_gated_layer_delta_cad' else ''
    out_path = PIPELINE_DIR / f'{args.model}_{dataset}_{args.method}_alpha{alpha_name}_tau{tau_name}_temp{temp_name}{layer_suffix}{limit_suffix}.json'
    save_json(result_obj, out_path)
    print(f'\n  Saved generation outputs → {out_path}')
    if csv_scores is None:
        print('  CSV skipped for baseline method')
    else:
        print(f'  trusted docs: {trusted_count}/{len(outputs)} ({trusted_rate:.2%})')
    print(f'  elapsed: {elapsed:.1f}s')
    return out_path

def main():
    parser = argparse.ArgumentParser(description='Step11: greedy baseline and CSV-gated pipeline generation')
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS + ['all'])
    parser.add_argument('--method', type=str, required=True, choices=METHODS)
    parser.add_argument('--alpha', type=float, default=1.0, help='CAD strength. For csv_adaptive_cad this is alpha_max.')
    parser.add_argument('--tau', type=float, default=0.5, help='CSV gate threshold for trusting a retrieved document.')
    parser.add_argument('--contrast_layer', type=int, default=23, help='Step10 hidden_states index for csv_gated_layer_delta_cad.')
    parser.add_argument('--context_ucd_beta', type=float, default=1.0, help='UCD path-history decay beta for Context-UCD-Energy; default 1.0.')
    parser.add_argument('--context_ucd_relative_top', type=float, default=0.0, help='Disabled for the main experiment. Must be 0.0.')
    parser.add_argument('--dola_layers', type=str, default=None, help="Official HF DoLA layers: 'low', 'high', or comma-separated layer list. Default: high.")
    parser.add_argument('--dola_repetition_penalty', type=float, default=1.2, help='Repetition penalty for official HF DoLA generation. Use 1.0 to disable.')
    parser.add_argument('--dola_mature_layer', type=int, default=-1, help='Legacy/no-op under official HF DoLA; final layer is handled internally by generate().')
    parser.add_argument('--dola_premature_layer', type=int, default=None, help='Legacy compatibility: converted to --dola_layers [layer] if --dola_layers is unset.')
    parser.add_argument('--dola_candidate_layers', type=str, default=None, help='Legacy compatibility: comma-separated list converted to --dola_layers if --dola_layers is unset.')
    parser.add_argument('--dola_relative_top', type=float, default=0.1, help='Legacy/no-op under official HF DoLA; kept only for old command compatibility.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for CSV validation/scoring. Generation is per-sample.')
    parser.add_argument('--max_new_tokens', type=int, default=20)
    parser.add_argument('--max_input_length', type=int, default=1024, help='Left-truncate prompts to this many tokens before generation. Use <=0 to disable.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Greedy decoding only. Must be 0.0 for this script.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Kept for compatibility; ignored under greedy decoding.')
    parser.add_argument('--stop_on_newline', action='store_true', default=True, help='Stop generation once the decoded generated text contains a newline. Default: True.')
    parser.add_argument('--no_stop_on_newline', action='store_false', dest='stop_on_newline')
    parser.add_argument('--validation_tolerance', type=float, default=0.002, help='Allowed absolute difference between recomputed eval AUROC and saved best AUROC.')
    parser.add_argument('--limit', type=int, default=None, help='Debug: only run the first N test samples. CSV validation is run only for CSV-gated methods.')
    args = parser.parse_args()
    if args.max_input_length is not None and args.max_input_length <= 0:
        args.max_input_length = None
    if args.temperature is not None and args.temperature > 0:
        raise ValueError('Step11 is configured for greedy decoding only. Use --temperature 0.0 and --top_p 1.0.')
    if args.method == 'context_ucd' and float(args.context_ucd_relative_top) != 0.0:
        raise ValueError('Context-UCD relative_top is disabled for the main experiment. Use --context_ucd_relative_top 0.0.')
    if args.method == 'dola' and (not args.dola_layers) and (args.dola_premature_layer is not None) and (args.dola_candidate_layers is not None):
        raise ValueError('For DoLA legacy arguments, use either --dola_premature_layer or --dola_candidate_layers, not both. Or set --dola_layers directly.')
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    print('\n' + '=' * 80)
    print('Step 11: CSV-gated Context-aware Decoding')
    print('=' * 80)
    print(f'Project root: {PROJECT_ROOT}')
    print(f'Model:        {args.model}')
    print(f'Dataset:      {args.dataset}')
    print(f'Method:       {args.method}')
    print(f'alpha:        {args.alpha}')
    print(f'tau:          {args.tau}')
    if args.method == 'csv_gated_layer_delta_cad':
        print(f'contrast L:   {args.contrast_layer}  (Step10 hidden_states index)')
    print(f'max_new:      {args.max_new_tokens}')
    print('decoding:     greedy (temperature=0.0; HF DoLA uses do_sample=False)')
    if args.method == 'context_ucd':
        print('Context-UCD:  UCD-energy, doc_prompt as base vs no_doc_prompt as amateur')
        print(f'Ctx-UCD beta decay: {args.context_ucd_beta}')
        print(f'Ctx-UCD rtop: {args.context_ucd_relative_top}')
    if args.method == 'dola':
        resolved = parse_dola_layers_config(args.dola_layers, args.dola_premature_layer, args.dola_candidate_layers)
        print(f'DoLA impl:    HuggingFace official generate(dola_layers=...)')
        print(f'DoLA layers:  {resolved}')
        print(f'DoLA rep_pen: {args.dola_repetition_penalty}')
        print(f'DoLA custom:  {DOLA_CUSTOM_GENERATE}')
        if args.dola_mature_layer != -1 or args.dola_relative_top != 0.1:
            print('DoLA note:    --dola_mature_layer/--dola_relative_top are legacy/no-op under official HF DoLA')
    datasets_to_run = DATASETS if args.dataset == 'all' else [args.dataset]
    needs_csv = args.method in CSV_METHODS
    print(f'needs_csv:    {needs_csv}')
    if args.method in BASELINE_METHODS:
        print('CSV scorer:   skipped for baseline method')
    else:
        print(f'CSV scorer:   required; reading retrieved-doc CSV from {CSV_DIR}')
    test_samples_by_dataset = {}
    for dataset in datasets_to_run:
        test_samples_by_dataset[dataset] = load_test_samples(dataset, limit=args.limit)
    ckpt_info = None
    validation = None
    test_scores_by_dataset = {dataset: None for dataset in datasets_to_run}
    if needs_csv:
        ckpt_info = resolve_best_csv_checkpoint(args.model)
        (score_model, score_tokenizer, _) = load_base_model_and_tokenizer(args.model)
        check_checkpoint_metadata(args.model, ckpt_info, score_model.config.hidden_size)
        inject_csv_into_model(score_model, ckpt_info)
        validation = validate_csv_full_eval(model=score_model, tokenizer=score_tokenizer, ckpt_info=ckpt_info, batch_size=args.batch_size, tolerance=args.validation_tolerance)
        print('\n=== Scoring test retrieved documents with retrieved-doc CSV ===')
        for dataset in datasets_to_run:
            samples = test_samples_by_dataset[dataset]
            prompts = prepare_doc_scoring_prompts(score_tokenizer, samples)
            scored = score_prompts_with_csv(model=score_model, prompts=prompts, batch_size=args.batch_size, pad_id=score_tokenizer.pad_token_id, cls_layer=int(ckpt_info.ckpt['cls_layer']), cos_temp=float(ckpt_info.ckpt['cos_temp']), centroids=ckpt_info.ckpt['centroids'])
            test_scores_by_dataset[dataset] = scored
            trusted = int((scored.scores >= args.tau).sum())
            print(f'  {dataset}: P(rel) mean={scored.scores.mean():.4f}, trusted@tau={trusted}/{len(scored.scores)} ({trusted / max(1, len(scored.scores)):.2%})')
        print('\n=== Releasing CSV scorer model before clean generation ===')
        del score_model
        torch.cuda.empty_cache()
    else:
        print('\n=== Skipping CSV checkpoint, CSV validation, and CSV test scoring ===')
    (gen_model, gen_tokenizer, _) = load_base_model_and_tokenizer(args.model)
    output_paths = []
    for dataset in datasets_to_run:
        out_path = run_generation_for_dataset(model=gen_model, tokenizer=gen_tokenizer, dataset=dataset, samples=test_samples_by_dataset[dataset], csv_scores=test_scores_by_dataset[dataset], args=args, ckpt_info=ckpt_info, validation=validation)
        output_paths.append(str(out_path))
    print('\n' + '=' * 80)
    print('Step11 done.')
    print('Outputs:')
    for p in output_paths:
        print(f'  {p}')
    print('=' * 80)
if __name__ == '__main__':
    main()
