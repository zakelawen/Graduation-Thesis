import argparse
import gc
import json
import os
import pickle
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d2rag.prompts import STAGE1_SYSTEM
from d2rag.utils import append_jsonl, load_jsonl, to_serializable
TRUTH_METHOD_NAMES = ['Entropy', 'SemanticEntropy', 'MARS', 'LARS', 'SentSAR', 'TokenSAR', 'Confidence', 'Inside', 'AttentionScore', 'EccentricityUncertainty', 'EccentricityConfidence', 'MatrixDegreeConfidence', 'MatrixDegreeUncertainty', 'SumEigenUncertainty', 'PTrue', 'VerbalizedConfidence', 'SelfDetection', 'DirectionalEntailmentGraph']
NLI_METHODS = {'SemanticEntropy', 'EccentricityUncertainty', 'EccentricityConfidence', 'MatrixDegreeConfidence', 'MatrixDegreeUncertainty', 'SumEigenUncertainty', 'SelfDetection', 'DirectionalEntailmentGraph'}

def make_safe_isotonic(ttlm):
    import numpy as np

    def clean(x):
        return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=1000000.0, neginf=-1000000.0)

    class SafeIsotonicRegression(ttlm.normalizers.IsotonicRegression):

        def fit(self, generation_performance_scores, truth_values):
            super().fit(generation_performance_scores, clean(truth_values))

        def __call__(self, truth_value):
            return super().__call__(float(clean(truth_value)))
    return SafeIsotonicRegression

def build_truth_methods(ttlm, nli_model, nli_tokenizer, aux_device):
    methods = []
    for name in TRUTH_METHOD_NAMES:
        cls = getattr(ttlm.truth_methods, name)
        if name in NLI_METHODS:
            methods.append(cls(model_for_entailment=nli_model, tokenizer_for_entailment=nli_tokenizer))
        elif name in ('MARS', 'LARS'):
            methods.append(cls(device=aux_device))
        else:
            methods.append(cls())
    return methods

def is_oom(e: Exception) -> bool:
    return 'out of memory' in str(e).lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--settings', default=os.path.join(os.path.dirname(__file__), '..', 'configs', 'settings.json'))
    ap.add_argument('--nli', default='microsoft/deberta-large-mnli')
    ap.add_argument('--aux-device', default='cuda:1')
    ap.add_argument('--nli-device', default='cuda:0')
    ap.add_argument('--size-of-data', type=int, default=10)
    ap.add_argument('--max-retry', type=int, default=2)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DebertaForSequenceClassification, DebertaTokenizer
    import TruthTorchLM as ttlm
    settings = json.load(open(args.settings, encoding='utf-8'))
    model_path = settings['models'].get(args.model, args.model)
    nli_model = DebertaForSequenceClassification.from_pretrained(args.nli).to(args.nli_device)
    nli_tokenizer = DebertaTokenizer.from_pretrained(args.nli)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto', attn_implementation='eager')
    model.config.output_attentions = True
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    truth_methods = build_truth_methods(ttlm, nli_model, nli_tokenizer, args.aux_device)
    norm_path = args.out + '.normalizers.pkl'
    safe_isotonic = make_safe_isotonic(ttlm)
    for m in truth_methods:
        m.set_normalizer(safe_isotonic())
    if os.path.exists(norm_path):
        for (m, iso) in zip(truth_methods, pickle.load(open(norm_path, 'rb'))):
            m.normalizer.iso_reg = iso
        print(f'loaded normalizers from {norm_path}')
    else:
        ttlm.calibrate_truth_method(dataset=load_jsonl(args.calib), model=model, truth_methods=truth_methods, tokenizer=tokenizer, correctness_evaluator=ttlm.evaluators.ModelJudge('gpt-4o-mini'), size_of_data=args.size_of_data, return_method_details=True, seed=0, max_new_tokens=100, wandb_push_method_details=False, pad_token_id=tokenizer.eos_token_id)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        pickle.dump([m.normalizer.iso_reg for m in truth_methods], open(norm_path, 'wb'))
        print(f'calibrated; normalizers saved to {norm_path}')
    test = load_jsonl(args.test)
    done = len(load_jsonl(args.out)) if os.path.exists(args.out) else 0
    skipped_path = args.out.replace('.jsonl', '_skipped.jsonl')
    print(f'{done}/{len(test)} already done')
    for idx in range(done, len(test)):
        sample = test[idx]
        record = {'idx': idx, 'question': sample['question'], 'ground_truth': sample['ground_truths'][0]}
        for attempt in range(args.max_retry + 1):
            try:
                out = ttlm.generate_with_truth_value(model=model, tokenizer=tokenizer, truth_methods=truth_methods, messages=[{'role': 'system', 'content': STAGE1_SYSTEM}, {'role': 'user', 'content': sample['question']}], max_new_tokens=50, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
                record['generated_text'] = out['generated_text']
                record['normalized_truth_values'] = to_serializable(out['normalized_truth_values'])
                record['unnormalized_truth_values'] = to_serializable(out.get('unnormalized_truth_values'))
                break
            except Exception as e:
                gc.collect()
                torch.cuda.empty_cache()
                if not is_oom(e) or attempt == args.max_retry:
                    record['error'] = f'{type(e).__name__}: {str(e)[:300]}'
                    append_jsonl(skipped_path, record)
                    break
                time.sleep(2)
        append_jsonl(args.out, record)
        gc.collect()
        torch.cuda.empty_cache()
        if (idx + 1) % 20 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {idx + 1}/{len(test)}", flush=True)
if __name__ == '__main__':
    main()
