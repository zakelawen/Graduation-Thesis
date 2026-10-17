import argparse
import gc
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d2rag.seper import EntailmentDeberta, calculate_uncertainty_soft_batch, create_collate_fn, gen_answers_batch, process_item_for_seper
from d2rag.utils import append_jsonl, load_jsonl
SEPER_KEYS = ['question', 'response_text', 'answers', 'likelihood', 'context_label', 'log_liks_agg', 'context']

class SampleGenerator:

    def __init__(self, model, tokenizer, max_new_tokens=20):
        (self.model, self.tokenizer, self.max_new_tokens) = (model, tokenizer, max_new_tokens)
        self.stop_sequences = ['\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:', tokenizer.eos_token]

    def batch_predict(self, prompt, num_generations, temperature, device='cuda', max_new_tokens=None, **_):
        import torch
        inputs = self.tokenizer([prompt] * num_generations, return_tensors='pt').to(device)
        inputs.pop('token_type_ids', None)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens, return_dict_in_generate=True, output_scores=True, temperature=temperature, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
        (answers, n_generated) = ([], [])
        n_in = inputs['input_ids'].shape[1]
        for seq in out.sequences:
            answer = self.tokenizer.decode(seq[n_in:], skip_special_tokens=False)
            for stop in self.stop_sequences:
                if stop and answer.endswith(stop):
                    answer = answer[:-len(stop)]
            answers.append(answer.strip())
            n_generated.append(len(self.tokenizer.encode(answer.strip())) or 1)
        scores = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        log_liks = [[s.item() for s in ts][:n] or [0.0] for (ts, n) in zip(scores, n_generated)]
        return (answers, log_liks, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--settings', default=os.path.join(os.path.dirname(__file__), '..', 'configs', 'settings.json'))
    ap.add_argument('--nli', default='microsoft/deberta-v2-xlarge-mnli')
    ap.add_argument('--nli-device', default='cuda:0')
    ap.add_argument('--num-generations', type=int, default=10)
    ap.add_argument('--sub-batch', type=int, default=10)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max-new-tokens', type=int, default=20)
    ap.add_argument('--max-context-words', type=int, default=20000)
    ap.add_argument('--chunk-size', type=int, default=8)
    ap.add_argument('--no-8bit', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    settings = json.load(open(args.settings, encoding='utf-8'))
    model_path = settings['models'].get(args.model, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
    quant = None if args.no_8bit else BitsAndBytesConfig(load_in_8bit=True)
    if quant is not None:
        import functools
        import transformers.modeling_utils as mu
        mu.dispatch_model = functools.partial(mu.dispatch_model, force_hooks=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant, device_map='auto', torch_dtype=torch.float16 if args.no_8bit else None).eval()
    generator = SampleGenerator(model, tokenizer, args.max_new_tokens)
    nli = EntailmentDeberta(args.nli, device=args.nli_device)
    collate = create_collate_fn(SEPER_KEYS)

    def seper_of(question, context, answers):
        example = {'question': question, 'context': context, 'answers': answers}
        example = gen_answers_batch(example, generator, args.temperature, args.num_generations, args.sub_batch, args.max_new_tokens, 'cuda', args.max_context_words)
        with torch.no_grad():
            return float(calculate_uncertainty_soft_batch(collate([process_item_for_seper(example)]), nli, args.chunk_size)[0])
    data = load_jsonl(args.input)
    if args.limit:
        data = data[:args.limit]
    done = len(load_jsonl(args.out)) if os.path.exists(args.out) else 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f'{done}/{len(data)} already done', flush=True)
    for idx in range(done, len(data)):
        item = data[idx]
        (question, answers, docs) = (item.get('question', ''), item.get('ground_truths', ['']), item.get('docs') or [])
        base = seper_of(question, '', answers)
        full = seper_of(question, '\n'.join((d.get('text', '') for d in docs)), answers)
        for doc in docs:
            doc['seper'] = seper_of(question, doc.get('text', ''), answers)
            doc['d_seper'] = doc['seper'] - base
        item.update(seper_baseline=base, seper_full=full, d_seper_full=full - base)
        append_jsonl(args.out, item)
        gc.collect()
        torch.cuda.empty_cache()
        if (idx + 1) % 20 == 0:
            print(f'  {idx + 1}/{len(data)}', flush=True)
if __name__ == '__main__':
    main()
