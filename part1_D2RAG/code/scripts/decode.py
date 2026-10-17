import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d2rag.prompts import build_stem, ctx_prompt, main_prompt
from d2rag.routing import choose_strategy
from d2rag.utils import load_jsonl, question_key, write_jsonl
MAX_NEW_TOKENS = 20

def alpha_of(strategy, alpha_pos, alpha_neg):
    return {'ctx-pos-acd': alpha_pos, 'ctx-neg-acd': alpha_neg}.get(strategy)

def last_position_head(torch, head):

    class LastPositionHead(torch.nn.Module):

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden_states):
            return self.inner(hidden_states[:, -1:, :])
    return LastPositionHead(head)

class Generator:

    def __init__(self, model_path):
        import torch
        from transformers import AutoTokenizer, GenerationConfig
        from d2rag.gen_wrapper import LMWrapper
        (self.torch, self.GenerationConfig) = (torch, GenerationConfig)
        self.model = LMWrapper(model_name_or_path=model_path, cache_dir='./cache', device_map='auto', max_memory=None, offload_folder='./offload')
        self.model.gen_model.lm_head = last_position_head(torch, self.model.gen_model.lm_head)
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = 'left'

    def _config(self, alpha_pos=None, alpha_neg=None):
        kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id, do_sample=False, return_dict_in_generate=True, output_scores=False)
        if alpha_pos is not None:
            kwargs['acd_alpha_pos'] = float(alpha_pos)
        if alpha_neg is not None:
            kwargs['acd_alpha_neg'] = float(alpha_neg)
        return self.GenerationConfig(**kwargs)

    def run(self, item, strategy, alpha):
        (stem, docs) = (build_stem(item), item.get('docs') or [])
        p_main = main_prompt(self.tok, stem)
        with self.torch.no_grad():
            if strategy in ('noctx-normal', 'ctx-normal'):
                prompt = p_main if strategy == 'noctx-normal' else ctx_prompt(self.tok, stem, docs, 'pos')[0]
                ids = self.tok(prompt, return_tensors='pt').to(self.model.device).input_ids
                out = self.model.generate_contrast(inputs=ids, generation_config=self._config(), mode=None)
                start = ids.shape[1]
            else:
                polarity = 'pos' if strategy == 'ctx-pos-acd' else 'neg'
                p_ctx = ctx_prompt(self.tok, stem, docs, polarity)[0]
                batch = self.tok([p_main, p_ctx], return_tensors='pt', padding='longest').to(self.model.device)
                (main_ids, ctx_ids) = (batch.input_ids[0:1], batch.input_ids[1:2])
                (main_mask, ctx_mask) = (batch.attention_mask[0:1], batch.attention_mask[1:2])
                extra = {'inputs_pos': ctx_ids, 'attention_mask_pos': ctx_mask} if polarity == 'pos' else {'inputs_neg': ctx_ids, 'attention_mask_neg': ctx_mask}
                cfg = self._config(alpha_pos=alpha if polarity == 'pos' else None, alpha_neg=alpha if polarity == 'neg' else None)
                out = self.model.generate_contrast(inputs=main_ids, attention_mask=main_mask, generation_config=cfg, mode='acd', **extra)
                start = main_ids.shape[1]
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return self.tok.decode(out.sequences[0][start:], skip_special_tokens=True).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--routed', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--settings', default=os.path.join(os.path.dirname(__file__), '..', 'configs', 'settings.json'))
    args = ap.parse_args()
    settings = json.load(open(args.settings, encoding='utf-8'))
    model_key = args.model if args.model in settings['models'] else os.path.basename(args.model)
    (harmful, useful) = settings['d_seper_thresholds'][args.dataset][model_key]
    data = load_jsonl(args.routed)
    gen = Generator(settings['models'].get(args.model, args.model))
    cache = {}
    name = f'{args.dataset}_{model_key}'
    for alpha_pos in settings['alpha_pos']:
        for alpha_neg in settings['alpha_neg']:
            rows = []
            for (idx, item) in enumerate(data):
                strategy = choose_strategy(item['needs_retrieval'], item.get('d_seper_full'), harmful, useful)
                alpha = alpha_of(strategy, alpha_pos, alpha_neg)
                key = (question_key(item['question']), strategy, alpha)
                if key not in cache:
                    cache[key] = gen.run(item, strategy, alpha)
                rows.append({'index': idx, 'question': item.get('question'), 'raw_question': item.get('raw_question'), 'ground_truths': item.get('ground_truths'), 'needs_retrieval': item['needs_retrieval'], 'clf_score': item.get('clf_score'), 'd_seper_full_score': item.get('d_seper_full'), 'combo_alpha_pos': alpha_pos, 'combo_alpha_neg': alpha_neg, 'decoding_strategy': strategy, 'acd_alpha_pos': alpha if strategy == 'ctx-pos-acd' else None, 'acd_alpha_neg': alpha if strategy == 'ctx-neg-acd' else None, 'model_output_raw': cache[key]})
            write_jsonl(os.path.join(args.out_dir, f'{name}_pos_{alpha_pos}_neg_{alpha_neg}.jsonl'), rows)
if __name__ == '__main__':
    main()
