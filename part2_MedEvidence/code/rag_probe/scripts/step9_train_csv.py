import sys
import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from csv_module.llm_layers import add_tsv_layers
from csv_module.train_utils import collate_fn, get_last_non_padded_token_rep, compute_ot_loss_cos, update_centroids_ema_hard
MODEL_REGISTRY = {'gemma2b': {'hf_name': os.environ.get('GEMMA_MODEL_PATH', '/home/qluai/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf')}, 'qwen3_4b': {'hf_name': os.environ.get('QWEN3_4B_MODEL_PATH', 'Qwen/Qwen3-4B-Base')}}
DATA_DIR = PROJECT_ROOT / 'data' / 'final'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'csv'
DEFAULTS = {'lam': 5.0, 'cos_temp': 0.1, 'ema_decay': 0.99, 'lr': 0.005, 'batch_size': 8, 'num_epochs': 20, 'patience': 5}

def extract_doc_text(doc) -> str:
    if isinstance(doc, dict):
        title = str(doc.get('title', '')).strip()
        text = str(doc.get('text', '')).strip()
        assert text, f"Document dict has empty 'text' field: {doc.keys()}"
        if title:
            return f'Title: {title}\nText: {text}'
        return text
    return str(doc).strip()

def setup_tokenizer(hf_name: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'  Set tokenizer.pad_token = eos_token (id={tokenizer.pad_token_id})')
    else:
        print(f"  tokenizer.pad_token already set: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    return tokenizer

def load_and_tokenize(split: str, tokenizer) -> tuple:
    path = DATA_DIR / f'{split}.json'
    with open(path, 'r') as f:
        data = json.load(f)
    prompts = []
    labels = []
    for item in data:
        q = item['question']
        doc_text = extract_doc_text(item['relevant_doc'])
        prompt_str = f'Document: {doc_text}\n\nQuestion: {q}\nAnswer:'
        tokens = tokenizer(prompt_str, return_tensors='pt').input_ids
        prompts.append(tokens)
        labels.append(1)
        doc_text = extract_doc_text(item['distracting_doc'])
        prompt_str = f'Document: {doc_text}\n\nQuestion: {q}\nAnswer:'
        tokens = tokenizer(prompt_str, return_tensors='pt').input_ids
        prompts.append(tokens)
        labels.append(0)
    print(f'  Loaded {split}: {len(data)} samples → {len(prompts)} prompts')
    return (prompts, labels)

@torch.no_grad()
def run_padding_sanity_check(model, tokenizer, pad_id: int):
    print('\n  Running right-padding sanity check …')
    device = next(model.parameters()).device
    short_prompt = 'Document: The sky is blue.\n\nQuestion: What color is the sky?\nAnswer:'
    long_prompt = 'Document: The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China.\n\nQuestion: What is the Great Wall of China?\nAnswer:'
    short_ids = tokenizer(short_prompt, return_tensors='pt').input_ids
    long_ids = tokenizer(long_prompt, return_tensors='pt').input_ids

    def _forward_single(ids):
        ids = ids.to(device)
        mask = torch.ones_like(ids, dtype=torch.long)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        return torch.stack([h[0, -1, :].cpu().float() for h in out.hidden_states], dim=0)
    (batch_p, _, batch_mask) = collate_fn([short_ids, long_ids], [0, 0], pad_id=pad_id)
    batch_p = batch_p.to(device)
    batch_mask = batch_mask.to(device)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        out = model.model(input_ids=batch_p, attention_mask=batch_mask, output_hidden_states=True)
    short_len = short_ids.size(1)
    long_len = long_ids.size(1)
    batched_short = torch.stack([h[0, short_len - 1, :].cpu().float() for h in out.hidden_states], dim=0)
    batched_long = torch.stack([h[1, long_len - 1, :].cpu().float() for h in out.hidden_states], dim=0)
    single_short = _forward_single(short_ids)
    single_long = _forward_single(long_ids)
    cos_short = F.cosine_similarity(single_short, batched_short, dim=1)
    cos_long = F.cosine_similarity(single_long, batched_long, dim=1)
    min_cos_short = cos_short.min().item()
    min_cos_long = cos_long.min().item()
    print(f'    Short prompt: min cosine similarity = {min_cos_short:.6f} (across {len(cos_short)} layers)')
    print(f'    Long prompt:  min cosine similarity = {min_cos_long:.6f} (across {len(cos_long)} layers)')
    threshold = 0.9999
    if min_cos_short < threshold or min_cos_long < threshold:
        raise RuntimeError(f'Right-padding sanity check FAILED.\n  Short prompt min cosine: {min_cos_short:.6f}\n  Long prompt min cosine:  {min_cos_long:.6f}\n  Threshold: {threshold}\nThis model may not correctly handle right-padding. Investigate transformers version or attention implementation before proceeding.')
    print('    ✓ Right-padding sanity check passed.')

def get_layer_rep(output, attention_mask, cls_layer: int):
    hidden_states = output.hidden_states
    if cls_layer == -1:
        hs = hidden_states[-1]
    else:
        hs = hidden_states[cls_layer + 1]
    return get_last_non_padded_token_rep(hs, attention_mask)

def parse_layer_list(layer_str: str, arg_name: str) -> list[int]:
    layers = []
    seen = set()
    for raw in layer_str.split(','):
        raw = raw.strip()
        if not raw:
            continue
        try:
            layer = int(raw)
        except ValueError as exc:
            raise ValueError(f'{arg_name} must be a comma-separated list of integers, got: {layer_str!r}') from exc
        if layer not in seen:
            layers.append(layer)
            seen.add(layer)
    if not layers:
        raise ValueError(f'{arg_name} is empty. Example: --str_layers 0,1,2,3,4,5')
    return layers

def train_single_config(model_name: str, hf_name: str, str_layer: int, cls_layer: int, train_prompts: list, train_labels: list, eval_prompts: list, eval_labels: list, pad_id: int, hparams: dict, device: torch.device, run_sanity_check: bool=False) -> dict:
    cls_label = 'final' if cls_layer == -1 else str(cls_layer)
    print(f'\n  --- inject=layer_{str_layer}, classify=layer_{cls_label} ---')
    model = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.float16, device_map='auto', low_cpu_mem_usage=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    if run_sanity_check:
        from transformers import AutoTokenizer as _AutoTok
        _tok = _AutoTok.from_pretrained(hf_name)
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token
        run_padding_sanity_check(model, _tok, pad_id)
    hidden_size = model.config.hidden_size
    tsv_param = nn.Parameter(torch.zeros(1, 1, hidden_size, dtype=torch.float32, device=device))
    alpha = [hparams['lam']]
    add_tsv_layers(model, tsv_param, alpha, str_layer, model_name)
    optimizer = torch.optim.AdamW([tsv_param], lr=hparams['lr'])
    batch_size = hparams['batch_size']
    num_epochs = hparams['num_epochs']
    cos_temp = hparams['cos_temp']
    ema_decay = hparams['ema_decay']
    patience = hparams['patience']
    centroids = torch.randn(2, hidden_size, dtype=torch.float32, device=device)
    centroids = F.normalize(centroids, p=2, dim=1)
    num_train = len(train_labels)
    best_auroc = -1.0
    best_acc = 0.0
    best_avg_margin = 0.0
    best_epoch = -1
    final_auroc = 0.0
    final_acc = 0.0
    final_avg_margin = 0.0
    no_improve = 0
    best_tsv = tsv_param.detach().cpu().clone()
    best_centroids = centroids.detach().cpu().clone()
    for epoch in range(num_epochs):
        running_loss = 0.0
        total = 0
        perm = torch.randperm(num_train).tolist()
        num_batches = (num_train + batch_size - 1) // batch_size
        next_progress_pct = 10
        for (batch_idx, batch_start) in enumerate(range(0, num_train, batch_size), start=1):
            idx = perm[batch_start:batch_start + batch_size]
            batch_p_list = [train_prompts[i] for i in idx]
            batch_l_list = [train_labels[i] for i in idx]
            (batch_p, batch_l, attention_mask) = collate_fn(batch_p_list, batch_l_list, pad_id=pad_id)
            batch_p = batch_p.to(device)
            batch_l = batch_l.to(device)
            attention_mask = attention_mask.to(device)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model.model(input_ids=batch_p, attention_mask=attention_mask, output_hidden_states=True)
                last_token_rep = get_layer_rep(output, attention_mask, cls_layer)
            last_token_rep_f = last_token_rep.float()
            batch_labels_oh = F.one_hot(batch_l, num_classes=2).float()
            (loss, _) = compute_ot_loss_cos(last_token_rep_f, centroids, batch_labels_oh, cos_temp)
            with torch.no_grad():
                centroids = update_centroids_ema_hard(centroids, last_token_rep_f, batch_labels_oh, ema_decay)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([tsv_param], max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item() * batch_l.size(0)
            total += batch_l.size(0)
            completed_pct = int(batch_idx * 100 / num_batches)
            while next_progress_pct <= 100 and completed_pct >= next_progress_pct:
                print(f'    Epoch {epoch + 1}/{num_epochs}: {batch_idx}/{num_batches} batches ({next_progress_pct}% complete)', flush=True)
                next_progress_pct += 10
        epoch_loss = running_loss / total
        metrics = evaluate(model, centroids, eval_prompts, eval_labels, device, batch_size, cls_layer, cos_temp, pad_id)
        auroc = metrics['auroc']
        acc = metrics['accuracy']
        avg_margin = metrics['avg_margin']
        final_auroc = auroc
        final_acc = acc
        final_avg_margin = avg_margin
        is_best = auroc > best_auroc
        if is_best:
            best_auroc = auroc
            best_acc = acc
            best_avg_margin = avg_margin
            best_epoch = epoch + 1
            best_tsv = tsv_param.detach().cpu().clone()
            best_centroids = centroids.detach().cpu().clone()
            no_improve = 0
        else:
            no_improve += 1
        print(f'    Epoch {epoch + 1:2d}/{num_epochs}: loss={epoch_loss:.4f}  eval_AUROC={auroc:.4f}  eval_Acc={acc:.4f}  eval_Margin={avg_margin:.4f}' + ('  ★ best' if is_best else ''))
        if no_improve >= patience:
            print(f'    Early stop: no improvement for {patience} epochs')
            break
    ckpt_path = RESULTS_DIR / f'{model_name}_inject_{str_layer}_cls_{cls_label}_csv.pt'
    torch.save({'tsv': best_tsv, 'centroids': best_centroids, 'str_layer': str_layer, 'cls_layer': cls_layer, 'lam': hparams['lam'], 'cos_temp': hparams['cos_temp'], 'best_auroc': float(best_auroc), 'best_accuracy': float(best_acc), 'best_avg_margin': float(best_avg_margin), 'best_epoch': int(best_epoch), 'model_name': model_name, 'hf_name': hf_name}, ckpt_path)
    print(f'    Saved checkpoint → {ckpt_path}')
    del model
    torch.cuda.empty_cache()
    return {'str_layer': str_layer, 'cls_layer': cls_layer, 'best_auroc': float(best_auroc), 'best_accuracy': float(best_acc), 'best_avg_margin': float(best_avg_margin), 'best_epoch': int(best_epoch), 'final_auroc': float(final_auroc), 'final_accuracy': float(final_acc), 'final_avg_margin': float(final_avg_margin), 'checkpoint': str(ckpt_path)}

@torch.no_grad()
def evaluate(model, centroids, eval_prompts, eval_labels, device, batch_size, cls_layer, cos_temp, pad_id) -> dict:
    model.eval()
    all_scores = []
    all_preds = []
    all_labels = []
    all_margins = []
    num_eval = len(eval_prompts)
    for batch_start in range(0, num_eval, batch_size):
        batch_p_list = eval_prompts[batch_start:batch_start + batch_size]
        batch_l_list = eval_labels[batch_start:batch_start + batch_size]
        (batch_p, batch_l, attention_mask) = collate_fn(batch_p_list, batch_l_list, pad_id=pad_id)
        batch_p = batch_p.to(device)
        attention_mask = attention_mask.to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = model.model(input_ids=batch_p, attention_mask=attention_mask, output_hidden_states=True)
            last_token_rep = get_layer_rep(output, attention_mask, cls_layer)
        last_token_rep = F.normalize(last_token_rep.float(), p=2, dim=-1)
        c = F.normalize(centroids, p=2, dim=-1)
        similarities = torch.matmul(last_token_rep, c.T)
        logits = similarities / cos_temp
        probs = torch.softmax(logits, dim=-1)
        scores = probs[:, 1]
        preds = torch.argmax(probs, dim=-1)
        margins = torch.abs(logits[:, 1] - logits[:, 0])
        all_scores.append(scores.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(torch.tensor(batch_l_list))
        all_margins.append(margins.cpu())
    all_scores = torch.cat(all_scores).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_margins = torch.cat(all_margins).numpy()
    return {'auroc': float(roc_auc_score(all_labels, all_scores)), 'accuracy': float(accuracy_score(all_labels, all_preds)), 'avg_margin': float(all_margins.mean())}

def main():
    parser = argparse.ArgumentParser(description='Step 9: Train CSV')
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--str_layer', type=int, default=None, help='Injection layer (transformer layer, 0-based)')
    parser.add_argument('--str_layers', type=str, default=None, help="Comma-separated injection layers for selected-layer grid search (e.g. '0,1,2,3,4,5'). Mutually exclusive with --str_layer and --sweep_layers.")
    parser.add_argument('--sweep_layers', action='store_true', help='Sweep all injection layers')
    parser.add_argument('--cls_layer', type=int, default=None, help='Classification layer (transformer layer, 0-based). Default: -1 (final). Must be > str_layer.')
    parser.add_argument('--cls_layers', type=str, default=None, help="Comma-separated classification layers for grid search (e.g. '3,5,8'). Used with --sweep_layers.")
    parser.add_argument('--lam', type=float, default=DEFAULTS['lam'])
    parser.add_argument('--cos_temp', type=float, default=DEFAULTS['cos_temp'])
    parser.add_argument('--ema_decay', type=float, default=DEFAULTS['ema_decay'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--num_epochs', type=int, default=DEFAULTS['num_epochs'])
    parser.add_argument('--patience', type=int, default=DEFAULTS['patience'], help='Early-stopping patience. Set to num_epochs to disable.')
    parser.add_argument('--skip_sanity_check', action='store_true', help='Skip the right-padding sanity check on the first config.')
    parser.add_argument('--resume', action='store_true', help='Skip completed configs if both the checkpoint .pt and result .json exist.')
    args = parser.parse_args()
    if args.str_layers is not None and args.sweep_layers:
        parser.error('Use either --str_layers or --sweep_layers, not both.')
    if args.str_layers is not None and args.str_layer is not None:
        parser.error('Use either --str_layers or --str_layer, not both.')
    if args.str_layer is None and args.str_layers is None and (not args.sweep_layers):
        parser.error('Must specify one of --str_layer, --str_layers, or --sweep_layers')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    model_key = args.model
    hf_name = MODEL_REGISTRY[model_key]['hf_name']
    hparams = {'lam': args.lam, 'cos_temp': args.cos_temp, 'ema_decay': args.ema_decay, 'lr': args.lr, 'batch_size': args.batch_size, 'num_epochs': args.num_epochs, 'patience': args.patience}
    print(f'Model: {model_key} ({hf_name})')
    print(f'Hyperparams: {hparams}')
    print('\nSetting up tokenizer …')
    tokenizer = setup_tokenizer(hf_name)
    pad_id = tokenizer.pad_token_id
    print('\nLoading data …')
    (train_prompts, train_labels) = load_and_tokenize('train', tokenizer)
    (eval_prompts, eval_labels) = load_and_tokenize('eval', tokenizer)
    config = AutoConfig.from_pretrained(hf_name)
    num_layers = config.num_hidden_layers
    try:
        if args.str_layers is not None:
            inject_layers = parse_layer_list(args.str_layers, '--str_layers')
        elif args.sweep_layers:
            inject_layers = list(range(num_layers))
        else:
            inject_layers = [args.str_layer]
        if args.cls_layers is not None:
            cls_layers = parse_layer_list(args.cls_layers, '--cls_layers')
        elif args.cls_layer is not None:
            cls_layers = [args.cls_layer]
        else:
            cls_layers = [-1]
    except ValueError as exc:
        parser.error(str(exc))
    invalid_inject = [x for x in inject_layers if x < 0 or x >= num_layers]
    if invalid_inject:
        parser.error(f'Invalid injection layer(s): {invalid_inject}. For {model_key}, valid injection layers are 0..{num_layers - 1}.')
    invalid_cls = [x for x in cls_layers if x != -1 and (x < 0 or x >= num_layers)]
    if invalid_cls:
        parser.error(f'Invalid classification layer(s): {invalid_cls}. For {model_key}, valid cls layers are 0..{num_layers - 1}, or -1 for final.')
    print(f'Injection layers: {inject_layers}')
    print(f'Classification layers: {cls_layers} (-1 = final)')
    all_results = {}
    first_config = True
    for inj in inject_layers:
        for cls in cls_layers:
            if cls != -1 and inj >= cls:
                print(f'  [skip] inject={inj} >= cls={cls}')
                continue
            cls_label = 'final' if cls == -1 else str(cls)
            key = f'inject_{inj}_cls_{cls_label}'
            result_path = RESULTS_DIR / f'{model_key}_{key}_csv_result.json'
            ckpt_path = RESULTS_DIR / f'{model_key}_inject_{inj}_cls_{cls_label}_csv.pt'
            if args.resume and result_path.exists() and ckpt_path.exists():
                try:
                    with open(result_path, 'r') as f:
                        result = json.load(f)
                    all_results[key] = result
                    print(f'  [resume skip] {key} already completed')
                    continue
                except json.JSONDecodeError:
                    print(f'  [resume warning] {result_path} is broken; rerunning {key}')
            run_sanity = first_config and (not args.skip_sanity_check)
            first_config = False
            result = train_single_config(model_name=model_key, hf_name=hf_name, str_layer=inj, cls_layer=cls, train_prompts=train_prompts, train_labels=train_labels, eval_prompts=eval_prompts, eval_labels=eval_labels, pad_id=pad_id, hparams=hparams, device=device, run_sanity_check=run_sanity)
            all_results[key] = result
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
    if not all_results:
        raise RuntimeError('No valid configs were run. Check layer choices: for non-final cls_layer, require inject_layer < cls_layer.')
    if len(all_results) > 1:
        sweep_path = RESULTS_DIR / f'{model_key}_csv_sweep.json'
        with open(sweep_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nSaved sweep → {sweep_path}')
        best_key = max(all_results, key=lambda k: all_results[k]['best_auroc'])
        print(f"Best: {best_key} AUROC={all_results[best_key]['best_auroc']:.4f}  Acc={all_results[best_key]['best_accuracy']:.4f}  Margin={all_results[best_key]['best_avg_margin']:.4f}")
        (fig, ax) = plt.subplots(figsize=(12, 5))
        for cls in cls_layers:
            cls_label = 'final' if cls == -1 else str(cls)
            (injs, aurocs) = ([], [])
            for inj in inject_layers:
                key = f'inject_{inj}_cls_{cls_label}'
                if key in all_results:
                    injs.append(inj)
                    aurocs.append(all_results[key]['best_auroc'])
            if injs:
                ax.plot(injs, aurocs, 'o-', linewidth=1.5, markersize=4, label=f'cls=layer_{cls_label}')
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Injection Layer (transformer layer index, 0-based)')
        ax.set_ylabel('Best AUROC')
        ax.set_title(f'{model_key} — CSV AUROC by Injection × Classification Layer')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = RESULTS_DIR / f'{model_key}_csv_auroc_by_layer.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f'Saved figure → {fig_path}')
    print('\nDone.')
if __name__ == '__main__':
    main()
