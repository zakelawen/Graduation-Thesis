import argparse
import json
import os
import sys
import random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
import step9_train_csv as base
DATA_DIR = PROJECT_ROOT / 'data' / 'final_retrieved'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'csv_retrieved'
FIXED_SEED = 42
MODEL_REGISTRY = {'gemma2b': {'hf_name': os.environ.get('GEMMA_MODEL_PATH', 'google/gemma-2-2b'), 'run_name': 'gemma2b_retrieved', 'ema_decay_default': 0.99, 'batch_size_default': 8}, 'gemma9b': {'hf_name': os.environ.get('GEMMA9B_MODEL_PATH', 'google/gemma-2-9b'), 'run_name': 'gemma9b_retrieved', 'ema_decay_default': 0.99, 'batch_size_default': 4}, 'qwen3_4b': {'hf_name': os.environ.get('QWEN3_4B_MODEL_PATH', 'Qwen/Qwen3-4B-Base'), 'run_name': 'qwen3_4b_retrieved', 'ema_decay_default': 0.999686, 'batch_size_default': 4}}
DEFAULTS = {'lam': 5.0, 'cos_temp': 0.1, 'lr': 0.005, 'num_epochs': 20, 'patience': 5}

def set_global_seed(seed: int=FIXED_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_tokenize_retrieved(split: str, tokenizer) -> tuple[list, list[int]]:
    path = DATA_DIR / f'{split}.json'
    if not path.exists():
        raise FileNotFoundError(f'Cannot find {path}. Run scripts/step4b_build_retrieved_dataset.py first.')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts = []
    labels = []
    label_counts = {0: 0, 1: 0}
    source_counts = {}
    for (idx, item) in enumerate(data):
        q = item['question']
        if 'doc' not in item:
            raise KeyError(f"Item {idx} has no 'doc' field. Keys: {list(item.keys())}")
        if 'label' in item:
            label = int(item['label'])
        else:
            label_name = str(item.get('label_name', '')).lower().strip()
            if label_name == 'relevant':
                label = 1
            elif label_name == 'distracting':
                label = 0
            else:
                raise ValueError(f'Item {idx} has invalid label_name={label_name!r}')
        if label not in (0, 1):
            raise ValueError(f'Item {idx} has invalid label={label!r}; expected 0/1')
        doc_text = base.extract_doc_text(item['doc'])
        prompt_str = f'Document: {doc_text}\n\nQuestion: {q}\nAnswer:'
        tokens = tokenizer(prompt_str, return_tensors='pt').input_ids
        prompts.append(tokens)
        labels.append(label)
        label_counts[label] = label_counts.get(label, 0) + 1
        src = str(item.get('source_dataset', 'unknown')).lower().strip()
        source_counts[src] = source_counts.get(src, 0) + 1
    print(f'  Loaded {split}: {len(data)} retrieved-doc samples → {len(prompts)} prompts')
    print(f'    labels: distracting={label_counts.get(0, 0)}  relevant={label_counts.get(1, 0)}')
    print(f'    sources: {source_counts}')
    return (prompts, labels)

def parse_args():
    parser = argparse.ArgumentParser(description='Step 9R: Train retrieved-document CSV scorer')
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--str_layer', type=int, default=None, help='Injection layer, 0-based transformer layer index')
    parser.add_argument('--str_layers', type=str, default=None, help='Comma-separated injection layers, e.g. 0,1,2,3,4,5')
    parser.add_argument('--sweep_layers', action='store_true', help='Sweep all transformer layers as injection layers')
    parser.add_argument('--cls_layer', type=int, default=None, help='Classification layer, 0-based; -1 = final')
    parser.add_argument('--cls_layers', type=str, default=None, help='Comma-separated classification layers, e.g. 2,4,8,-1')
    parser.add_argument('--lam', type=float, default=DEFAULTS['lam'])
    parser.add_argument('--cos_temp', type=float, default=DEFAULTS['cos_temp'])
    parser.add_argument('--ema_decay', type=float, default=None, help='If omitted, uses model-specific default')
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    parser.add_argument('--batch_size', type=int, default=None, help='If omitted, uses model-specific default')
    parser.add_argument('--num_epochs', type=int, default=DEFAULTS['num_epochs'])
    parser.add_argument('--patience', type=int, default=DEFAULTS['patience'])
    parser.add_argument('--skip_sanity_check', action='store_true', help='Skip right-padding sanity check on first trained config')
    parser.add_argument('--resume', action='store_true', help='Skip configs with both result json and checkpoint pt')
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR), help='Default: data/final_retrieved')
    parser.add_argument('--results_dir', type=str, default=str(RESULTS_DIR), help='Default: results/csv_retrieved')
    args = parser.parse_args()
    if args.str_layers is not None and args.sweep_layers:
        parser.error('Use either --str_layers or --sweep_layers, not both.')
    if args.str_layers is not None and args.str_layer is not None:
        parser.error('Use either --str_layers or --str_layer, not both.')
    if args.str_layer is None and args.str_layers is None and (not args.sweep_layers):
        parser.error('Must specify one of --str_layer, --str_layers, or --sweep_layers')
    return args

def main():
    args = parse_args()
    set_global_seed(FIXED_SEED)
    print(f'Fixed random seed: {FIXED_SEED}')
    global DATA_DIR, RESULTS_DIR
    DATA_DIR = Path(args.data_dir).resolve()
    if not DATA_DIR.exists():
        raise FileNotFoundError(f'DATA_DIR does not exist: {DATA_DIR}. Run Step4b first.')
    RESULTS_DIR = Path(args.results_dir).resolve()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base.RESULTS_DIR = RESULTS_DIR
    model_key = args.model
    info = MODEL_REGISTRY[model_key]
    hf_name = info['hf_name']
    run_name = info['run_name']
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = int(info['batch_size_default'])
    ema_decay = args.ema_decay
    if ema_decay is None:
        ema_decay = float(info['ema_decay_default'])
    hparams = {'lam': args.lam, 'cos_temp': args.cos_temp, 'ema_decay': ema_decay, 'lr': args.lr, 'batch_size': batch_size, 'num_epochs': args.num_epochs, 'patience': args.patience}
    print('=== Step9R: Retrieved-doc CSV training ===')
    print(f'Base model: {model_key} ({hf_name})')
    print(f'Run name:   {run_name}')
    print(f'DATA_DIR:   {DATA_DIR}')
    print(f'RESULTS:    {RESULTS_DIR}')
    print(f'Hyperparams: {hparams}')
    print('\nSetting up tokenizer …')
    tokenizer = base.setup_tokenizer(hf_name)
    pad_id = tokenizer.pad_token_id
    print('\nLoading retrieved-doc data …')
    (train_prompts, train_labels) = load_and_tokenize_retrieved('train', tokenizer)
    (eval_prompts, eval_labels) = load_and_tokenize_retrieved('eval', tokenizer)
    config = AutoConfig.from_pretrained(hf_name)
    num_layers = config.num_hidden_layers
    try:
        if args.str_layers is not None:
            inject_layers = base.parse_layer_list(args.str_layers, '--str_layers')
        elif args.sweep_layers:
            inject_layers = list(range(num_layers))
        else:
            inject_layers = [args.str_layer]
        if args.cls_layers is not None:
            cls_layers = base.parse_layer_list(args.cls_layers, '--cls_layers')
        elif args.cls_layer is not None:
            cls_layers = [args.cls_layer]
        else:
            cls_layers = [-1]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    invalid_inject = [x for x in inject_layers if x < 0 or x >= num_layers]
    if invalid_inject:
        raise ValueError(f'Invalid injection layer(s): {invalid_inject}. For {model_key}, valid injection layers are 0..{num_layers - 1}.')
    invalid_cls = [x for x in cls_layers if x != -1 and (x < 0 or x >= num_layers)]
    if invalid_cls:
        raise ValueError(f'Invalid classification layer(s): {invalid_cls}. For {model_key}, valid cls layers are 0..{num_layers - 1}, or -1.')
    print(f'Injection layers: {inject_layers}')
    print(f'Classification layers: {cls_layers} (-1 = final)')
    device = torch.device('cuda')
    all_results = {}
    first_config = True
    for inj in inject_layers:
        for cls in cls_layers:
            if cls != -1 and inj >= cls:
                print(f'  [skip] inject={inj} >= cls={cls}')
                continue
            cls_label = 'final' if cls == -1 else str(cls)
            key = f'inject_{inj}_cls_{cls_label}'
            result_path = RESULTS_DIR / f'{run_name}_{key}_csv_result.json'
            ckpt_path = RESULTS_DIR / f'{run_name}_inject_{inj}_cls_{cls_label}_csv.pt'
            if args.resume and result_path.exists() and ckpt_path.exists():
                try:
                    with open(result_path, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    all_results[key] = result
                    print(f'  [resume skip] {key} already completed')
                    continue
                except json.JSONDecodeError:
                    print(f'  [resume warning] {result_path} is broken; rerunning {key}')
            run_sanity = first_config and (not args.skip_sanity_check)
            first_config = False
            set_global_seed(FIXED_SEED)
            print(f'  Fixed seed for this config: {FIXED_SEED}')
            result = base.train_single_config(model_name=run_name, hf_name=hf_name, str_layer=inj, cls_layer=cls, train_prompts=train_prompts, train_labels=train_labels, eval_prompts=eval_prompts, eval_labels=eval_labels, pad_id=pad_id, hparams=hparams, device=device, run_sanity_check=run_sanity)
            result['base_model_key'] = model_key
            result['run_name'] = run_name
            result['data_dir'] = str(DATA_DIR)
            result['dataset_variant'] = 'retrieved'
            result['fixed_seed'] = FIXED_SEED
            all_results[key] = result
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
    if not all_results:
        raise RuntimeError('No valid configs were run. Check layer choices: for non-final cls_layer, require inject_layer < cls_layer.')
    if len(all_results) > 1:
        sweep_path = RESULTS_DIR / f'{run_name}_csv_sweep.json'
        with open(sweep_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f'\nSaved retrieved sweep → {sweep_path}')
        best_key = max(all_results, key=lambda k: all_results[k]['best_auroc'])
        print(f"Best: {best_key} AUROC={all_results[best_key]['best_auroc']:.4f}  Acc={all_results[best_key]['best_accuracy']:.4f}  Margin={all_results[best_key]['best_avg_margin']:.4f}")
    else:
        only_key = next(iter(all_results.keys()))
        print(f'\nSingle config finished: {only_key}')
    print('\nDone.')
if __name__ == '__main__':
    main()
