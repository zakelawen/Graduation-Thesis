import argparse
import glob
import json
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d2rag.prompts import build_stem, ctx_prompt, main_prompt
from d2rag.routing import choose_strategy
from d2rag.utils import load_jsonl, question_key, write_jsonl
MAX_NEW_TOKENS = 20

def alpha_of(strategy, alpha_pos, alpha_neg):
    return {'ctx-pos-acd': alpha_pos, 'ctx-neg-acd': alpha_neg}.get(strategy)

def load_reuse(patterns, skip_strategies=()):
    cache = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            for rec in load_jsonl(path):
                strategy = rec.get('decoding_strategy') or rec.get('strategy')
                if strategy in skip_strategies:
                    continue
                alpha = rec.get('acd_alpha_pos') if strategy == 'ctx-pos-acd' else rec.get('acd_alpha_neg') if strategy == 'ctx-neg-acd' else None
                key = (question_key(rec.get('question')), strategy, alpha)
                if key not in cache or cache[key][1] in (None, -1):
                    cache[key] = (rec.get('model_output_raw'), rec.get('is_match'))
    return cache

def last_position_head(torch, head):

    class LastPositionHead(torch.nn.Module):

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden_states):
            return self.inner(hidden_states[:, -1:, :])
    return LastPositionHead(head)

class Generator:

    def __init__(self, model_path, full_logits=False):
        import torch
        from transformers import AutoTokenizer, GenerationConfig
        from d2rag.gen_wrapper import LMWrapper
        (self.torch, self.GenerationConfig) = (torch, GenerationConfig)
        self.model = LMWrapper(model_name_or_path=model_path, cache_dir='./cache', device_map='auto', max_memory=None, offload_folder='./offload')
        self.full_head = self.model.gen_model.lm_head
        self.last_head = last_position_head(torch, self.full_head)
        self.set_full_logits(full_logits)
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = 'left'

    def set_full_logits(self, full):
        self.model.gen_model.lm_head = self.full_head if full else self.last_head

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
    ap.add_argument('--model', required=True, help='llama / qwen 或模型路径')
    ap.add_argument('--dataset', required=True, help='用于读取 settings.json 里的 d_seper 阈值')
    ap.add_argument('--routed', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--reuse', nargs='*', default=[])
    ap.add_argument('--no-reuse-strategy', nargs='*', default=[], help='这些策略的旧输出不复用（例如修复 Llama 对比解码后：ctx-pos-acd ctx-neg-acd）')
    ap.add_argument('--dry-run', action='store_true', help='只统计需要生成多少条，不加载模型')
    ap.add_argument('--verify', type=int, default=0, help='抽查 N 条已有输出能否被原样复现')
    ap.add_argument('--check-head', type=int, default=0, help='抽查 N 条对比解码输出：全位置 logits 与只算最后位置的结果是否一致')
    ap.add_argument('--check-only', action='store_true', help='只做 --verify / --check-head 抽查，不生成')
    ap.add_argument('--all-ctx-normal', action='store_true', help='消融用：不判断是否检索、不做对比解码，所有题都带检索文档普通解码')
    ap.add_argument('--no-stage1', action='store_true', help='消融用：去掉第一阶段，所有题都检索，再按 d_seper 选择对比解码 / 普通解码')
    ap.add_argument('--settings', default=os.path.join(os.path.dirname(__file__), '..', 'configs', 'settings.json'))
    args = ap.parse_args()
    settings = json.load(open(args.settings, encoding='utf-8'))
    model_key = args.model if args.model in settings['models'] else os.path.basename(args.model)
    (harmful, useful) = settings['d_seper_thresholds'][args.dataset][model_key]
    data = load_jsonl(args.routed)

    def strategy_of(item):
        if args.all_ctx_normal:
            return 'ctx-normal'
        needs_ret = 1 if args.no_stage1 else item['needs_retrieval']
        return choose_strategy(needs_ret, item.get('d_seper_full'), harmful, useful)
    cache = load_reuse(args.reuse, set(args.no_reuse_strategy))
    print(f'{len(data)} items, {len(cache)} reusable outputs, thresholds harmful={harmful} useful={useful}')
    plan = {}
    for alpha_pos in settings['alpha_pos']:
        for alpha_neg in settings['alpha_neg']:
            for item in data:
                strategy = strategy_of(item)
                key = (question_key(item['question']), strategy, alpha_of(strategy, alpha_pos, alpha_neg))
                if key not in cache:
                    plan[key] = item
    by_strategy = {}
    for (_, strategy, _) in plan:
        by_strategy[strategy] = by_strategy.get(strategy, 0) + 1
    print(f'need to generate {len(plan)} outputs {by_strategy}')
    if args.dry_run:
        return
    gen = Generator(settings['models'].get(args.model, args.model)) if plan or args.verify or args.check_head else None
    if args.verify:
        items = {question_key(it['question']): it for it in data}
        per_strategy = {}
        for k in cache:
            if k[0] in items and len(per_strategy.setdefault(k[1], [])) < max(1, args.verify // 4):
                per_strategy[k[1]].append(k)
        for (strategy, keys) in per_strategy.items():
            same = sum((gen.run(items[k[0]], k[1], k[2]) == (cache[k][0] or '').strip() for k in keys))
            print(f'verify {strategy}: {same}/{len(keys)} regenerated outputs identical to the reused ones', flush=True)
    if args.check_head:
        acd = [(k, it) for (k, it) in plan.items() if k[1] in ('ctx-pos-acd', 'ctx-neg-acd')]
        acd.sort(key=lambda kv: sum((len(d.get('text') or '') for d in kv[1].get('docs') or [])))
        same = 0
        for ((q, strategy, alpha), item) in acd[:args.check_head]:
            gen.set_full_logits(True)
            full = gen.run(item, strategy, alpha)
            gen.set_full_logits(False)
            last = gen.run(item, strategy, alpha)
            same += full == last
            if full != last:
                print(f'  differs ({strategy}, alpha={alpha}): full={full!r} last={last!r}')
        print(f'check-head: {same}/{min(args.check_head, len(acd))} identical with full vs last-position logits', flush=True)
    if args.check_only:
        return
    if plan:
        t0 = time.time()
        for (i, ((q, strategy, alpha), item)) in enumerate(plan.items(), 1):
            cache[q, strategy, alpha] = (gen.run(item, strategy, alpha), None)
            if i % 50 == 0 or i == len(plan):
                print(f'  generated {i}/{len(plan)}  ({time.time() - t0:.0f}s)', flush=True)
    name = f'{args.dataset}_{model_key}'
    for alpha_pos in settings['alpha_pos']:
        for alpha_neg in settings['alpha_neg']:
            rows = []
            for (idx, item) in enumerate(data):
                strategy = strategy_of(item)
                alpha = alpha_of(strategy, alpha_pos, alpha_neg)
                (output, is_match) = cache[question_key(item['question']), strategy, alpha]
                row = {'index': idx, 'question': item.get('question'), 'raw_question': item.get('raw_question'), 'ground_truths': item.get('ground_truths'), 'needs_retrieval': item['needs_retrieval'], 'clf_score': item.get('clf_score'), 'd_seper_full_score': item.get('d_seper_full'), 'combo_alpha_pos': alpha_pos, 'combo_alpha_neg': alpha_neg, 'decoding_strategy': strategy, 'acd_alpha_pos': alpha if strategy == 'ctx-pos-acd' else None, 'acd_alpha_neg': alpha if strategy == 'ctx-neg-acd' else None, 'model_output_raw': output}
                if is_match is not None:
                    row['is_match'] = is_match
                rows.append(row)
            write_jsonl(os.path.join(args.out_dir, f'{name}_pos_{alpha_pos}_neg_{alpha_neg}.jsonl'), rows)
    print(f"wrote {len(settings['alpha_pos']) * len(settings['alpha_neg'])} files to {args.out_dir}")
if __name__ == '__main__':
    main()
