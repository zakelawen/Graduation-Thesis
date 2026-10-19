from __future__ import annotations
import argparse
import concurrent.futures
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REBUTTAL_ROOT = PROJECT_ROOT / 'Rebuttal'
sys.path.insert(0, str(REBUTTAL_ROOT / 'scripts'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
sys.path.insert(0, str(PROJECT_ROOT))
from rebuttal_common import extract_doc_text
from step12_evaluate_pipeline import get_answers, normalize_answer
SELECTORS = ('dpr', 'answer_oracle', 'csv', 'qwen36', 'qwen25', 'dus', 'cross_encoder', 'crag')
FENCE_RE = re.compile('^```(?:json)?\\s*(.*?)\\s*```$', re.DOTALL)
DEFAULT_QWEN25 = '/home/qluai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
QWEN25_SYSTEM = 'You are a strict evidence assessor for question answering. Judge the UTILITY of a passage: whether it contains sufficient evidence to help produce the correct answer to the question. Utility is stronger than topical relevance: a passage can be about the same topic yet lack the evidence needed to answer. Answer with a single word: Yes or No.'
QWEN25_USER_TEMPLATE = 'Question: {question}\n\nPassage:\n{passage}\n\nDoes this passage have utility for answering the question (i.e., does it contain evidence sufficient to derive the correct answer)? Answer Yes or No.'

def load_json(path: Path) -> Any:
    with path.open(encoding='utf-8') as f:
        return json.load(f)

def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, required=True, help='Top-k retrieval file from prepare_top10_retrieval.py')
    p.add_argument('--selector', required=True, choices=SELECTORS)
    p.add_argument('--dataset', default=None, help='Override dataset name in output meta')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--csv-model', default='qwen3_4b', choices=('gemma2b', 'gemma9b', 'qwen3_4b'))
    p.add_argument('--csv-batch-size', type=int, default=8)
    p.add_argument('--csv-validation-tolerance', type=float, default=0.002)
    p.add_argument('--skip-csv-validation', action='store_true')
    p.add_argument('--qwen-base-url', default='http://127.0.0.1:18000/v1')
    p.add_argument('--qwen-api-key', default='1')
    p.add_argument('--qwen-model', default='qwen-judge')
    p.add_argument('--qwen-prompt', type=Path, default=REBUTTAL_ROOT / 'prompts' / 'external_judge_utility_no_answer.json')
    p.add_argument('--qwen-workers', type=int, default=8)
    p.add_argument('--qwen-timeout', type=float, default=600.0)
    p.add_argument('--qwen-retries', type=int, default=2)
    p.add_argument('--qwen25-model', default=DEFAULT_QWEN25)
    p.add_argument('--qwen25-device', default='cuda:0')
    p.add_argument('--qwen25-batch-size', type=int, default=8)
    p.add_argument('--qwen25-max-len', type=int, default=1024)
    p.add_argument('--dus-selector-path', default=None)
    p.add_argument('--dus-device', default='cuda:0')
    p.add_argument('--dus-batch-size', type=int, default=8)
    p.add_argument('--dus-num-samples', type=int, default=1)
    p.add_argument('--dus-temperature', type=float, default=0.0)
    p.add_argument('--dus-max-new-tokens', type=int, default=256)
    p.add_argument('--dus-max-len', type=int, default=2048)
    p.add_argument('--cross-encoder-model', default='cross-encoder/ms-marco-MiniLM-L6-v2')
    p.add_argument('--cross-encoder-batch-size', type=int, default=32)
    p.add_argument('--crag-model', default=str(PROJECT_ROOT / 'CRAG/model'))
    p.add_argument('--crag-device', default='cuda:0')
    p.add_argument('--crag-batch-size', type=int, default=64)
    return p.parse_args()

def load_topk(path: Path, limit: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = load_json(path)
    if isinstance(obj, dict) and 'data' in obj:
        meta = obj.get('meta', {}) if isinstance(obj.get('meta'), dict) else {}
        rows = obj['data']
    elif isinstance(obj, list):
        meta = {}
        rows = obj
    else:
        raise ValueError(f'Bad top-k file format: {path}')
    if limit is not None:
        rows = rows[:limit]
    return (meta, rows)

def flatten_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for (qidx, row) in enumerate(rows):
        docs = row.get('retrieved_docs') or []
        if not docs and row.get('retrieved_doc') is not None:
            docs = [row['retrieved_doc']]
        for (rank, doc) in enumerate(docs):
            flat.append({'qidx': qidx, 'rank': rank, 'question': row['question'], 'answers': row.get('answers', []), 'doc': doc, 'doc_text': extract_doc_text(doc)})
    return flat

def answer_oracle_score(question: str, answers: Any, doc_text: str) -> float:
    norm_doc = normalize_answer(doc_text)
    for ans in get_answers({'answers': answers}):
        norm_ans = normalize_answer(ans)
        if norm_ans and norm_ans in norm_doc:
            return 1.0
    return 0.0

def score_dpr(flat: list[dict[str, Any]]) -> list[float]:
    return [float((x['doc'] or {}).get('score', 0.0)) for x in flat]

def score_answer_oracle(flat: list[dict[str, Any]]) -> list[float]:
    return [answer_oracle_score(x['question'], x['answers'], x['doc_text']) for x in flat]

def score_csv(flat: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[float], dict[str, Any]]:
    from step11_contrastive_decoding import check_checkpoint_metadata, inject_csv_into_model, load_base_model_and_tokenizer, prepare_doc_scoring_prompts, resolve_best_csv_checkpoint, score_prompts_with_csv, validate_csv_full_eval
    ckpt_info = resolve_best_csv_checkpoint(args.csv_model)
    (model, tokenizer, _) = load_base_model_and_tokenizer(args.csv_model)
    check_checkpoint_metadata(args.csv_model, ckpt_info, model.config.hidden_size)
    inject_csv_into_model(model, ckpt_info)
    validation = None
    if not args.skip_csv_validation:
        validation = validate_csv_full_eval(model=model, tokenizer=tokenizer, ckpt_info=ckpt_info, batch_size=args.csv_batch_size, tolerance=args.csv_validation_tolerance)
    samples = [{'question': x['question'], 'retrieved_doc': x['doc']} for x in flat]
    prompts = prepare_doc_scoring_prompts(tokenizer, samples)
    scored = score_prompts_with_csv(model=model, prompts=prompts, batch_size=args.csv_batch_size, pad_id=tokenizer.pad_token_id, cls_layer=int(ckpt_info.ckpt['cls_layer']), cos_temp=float(ckpt_info.ckpt['cos_temp']), centroids=ckpt_info.ckpt['centroids'])
    details = {'csv_model': args.csv_model, 'csv_best_key': ckpt_info.best_key, 'csv_checkpoint': str(ckpt_info.ckpt_path), 'csv_validation': validation}
    return ([float(x) for x in scored.scores.tolist()], details)

def load_qwen_prompt(path: Path) -> dict[str, Any]:
    return load_json(path)

def post_chat(base_url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(base_url.rstrip('/') + '/chat/completions', data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))

def parse_judge_json(content: str) -> dict[str, Any] | None:
    text = (content or '').strip()
    m = FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        (start, end) = (text.find('{'), text.rfind('}'))
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None

def qwen_score_from_obj(obj: dict[str, Any] | None) -> float:
    if not obj:
        return 0.5
    label = obj.get('label')
    conf = obj.get('confidence', 1.0)
    conf = max(0.0, min(1.0, float(conf))) if isinstance(conf, (int, float)) else 1.0
    return conf if label == 'supporting' else 1.0 - conf

def score_one_qwen(index: int, record: dict[str, Any], prompt: dict[str, Any], args: argparse.Namespace) -> tuple[int, float, dict[str, Any] | None, str | None]:
    payload = {'model': args.qwen_model, 'messages': [{'role': 'system', 'content': prompt['system']}, {'role': 'user', 'content': prompt['user_template'].format(question=record['question'], document=record['doc_text'])}], 'temperature': 0.0, 'max_tokens': 320, 'n': 1}
    if prompt.get('response_format'):
        payload['response_format'] = prompt['response_format']
    last_error = None
    for attempt in range(args.qwen_retries + 1):
        try:
            resp = post_chat(args.qwen_base_url, args.qwen_api_key, payload, args.qwen_timeout)
            content = ((resp.get('choices') or [{}])[0].get('message') or {}).get('content', '')
            obj = parse_judge_json(content)
            return (index, qwen_score_from_obj(obj), obj, None)
        except Exception as exc:
            last_error = repr(exc)
            if attempt < args.qwen_retries:
                time.sleep(2.0 * (attempt + 1))
    return (index, 0.5, None, last_error)

def score_qwen36(flat: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[float], dict[str, Any]]:
    prompt = load_qwen_prompt(args.qwen_prompt)
    scores = [0.5] * len(flat)
    parsed: list[Any] = [None] * len(flat)
    errors: list[Any] = [None] * len(flat)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.qwen_workers) as ex:
        futures = [ex.submit(score_one_qwen, i, rec, prompt, args) for (i, rec) in enumerate(flat)]
        for (done, fut) in enumerate(concurrent.futures.as_completed(futures), start=1):
            (idx, score, obj, err) = fut.result()
            scores[idx] = float(score)
            parsed[idx] = obj
            errors[idx] = err
            if done % 50 == 0 or done == len(futures):
                print(f'  qwen36 scored {done}/{len(futures)}')
    return (scores, {'parsed': parsed, 'errors': errors, 'qwen_model': args.qwen_model, 'qwen_base_url': args.qwen_base_url})

def first_token_ids(tokenizer: Any, words: list[str]) -> list[int]:
    ids = set()
    for word in words:
        for variant in (word, ' ' + word):
            toks = tokenizer(variant, add_special_tokens=False).input_ids
            if toks:
                ids.add(int(toks[0]))
    return list(ids)

def score_qwen25(flat: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[float], dict[str, Any]]:
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device(args.qwen25_device if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.qwen25_model, local_files_only=True, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.qwen25_model, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True).to(device)
    model.eval()
    yes_ids = torch.tensor(first_token_ids(tokenizer, ['Yes', 'yes', 'YES']), device=device)
    no_ids = torch.tensor(first_token_ids(tokenizer, ['No', 'no', 'NO']), device=device)
    prompts = []
    for rec in flat:
        messages = [{'role': 'system', 'content': QWEN25_SYSTEM}, {'role': 'user', 'content': QWEN25_USER_TEMPLATE.format(question=rec['question'], passage=rec['doc_text'])}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    scores: list[float] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(prompts), args.qwen25_batch_size), desc='qwen25_judge'):
            batch = prompts[start:start + args.qwen25_batch_size]
            encoded = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=args.qwen25_max_len).to(device)
            logits = model(**encoded).logits[:, -1, :]
            logp = torch.log_softmax(logits.float(), dim=-1)
            lse_yes = torch.logsumexp(logp[:, yes_ids], dim=-1)
            lse_no = torch.logsumexp(logp[:, no_ids], dim=-1)
            scores.extend(torch.sigmoid(lse_yes - lse_no).detach().cpu().numpy().astype(float).tolist())
    return (scores, {'model': 'Qwen2.5-7B-Instruct', 'model_path': args.qwen25_model, 'device': str(device), 'batch_size': args.qwen25_batch_size, 'max_len': args.qwen25_max_len})

def score_dus(flat: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[float], dict[str, Any]]:
    import torch
    from dus_utilityqwen_baseline import DEFAULT_SELECTOR, load_selector, score_records

    class DusArgs:
        pass
    dus_args = DusArgs()
    dus_args.batch_size = args.dus_batch_size
    dus_args.num_samples = args.dus_num_samples
    dus_args.temperature = args.dus_temperature
    dus_args.max_new_tokens = args.dus_max_new_tokens
    dus_args.max_len = args.dus_max_len
    selector_path = args.dus_selector_path or DEFAULT_SELECTOR
    device = torch.device(args.dus_device if torch.cuda.is_available() else 'cpu')
    (tokenizer, model) = load_selector(selector_path, device)
    records = [{'question': x['question'], 'doc_text': x['doc_text']} for x in flat]
    scores = score_records(records, tokenizer, model, device, dus_args)
    return ([float(x) for x in scores], {'selector_path': selector_path, 'device': str(device)})

def score_cross_encoder(flat: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[float], dict[str, Any]]:
    pairs = [(x['question'], x['doc_text']) for x in flat]
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(args.cross_encoder_model)
        scores = model.predict(pairs, batch_size=args.cross_encoder_batch_size, show_progress_bar=True)
        return ([float(x) for x in np.asarray(scores).reshape(-1).tolist()], {'model': args.cross_encoder_model, 'backend': 'sentence_transformers'})
    except ModuleNotFoundError:
        import torch
        from tqdm.auto import tqdm
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(args.cross_encoder_model, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.cross_encoder_model, local_files_only=True).to(device)
        model.eval()
        scores: list[float] = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(pairs), args.cross_encoder_batch_size), desc='cross_encoder'):
                batch = pairs[start:start + args.cross_encoder_batch_size]
                encoded = tokenizer([q for (q, _) in batch], [d for (_, d) in batch], padding=True, truncation=True, return_tensors='pt').to(device)
                logits = model(**encoded).logits.detach().float().cpu().numpy()
                scores.extend((float(x) for x in np.asarray(logits).reshape(len(batch), -1)[:, 0].tolist()))
        return (scores, {'model': args.cross_encoder_model, 'backend': 'transformers', 'device': str(device)})

def score_crag(flat: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[float], dict[str, Any]]:
    import torch
    from tqdm.auto import tqdm
    from transformers import T5ForSequenceClassification, T5Tokenizer
    device = torch.device(args.crag_device if torch.cuda.is_available() else 'cpu')
    tokenizer = T5Tokenizer.from_pretrained(args.crag_model, local_files_only=True)
    model = T5ForSequenceClassification.from_pretrained(args.crag_model, num_labels=1, local_files_only=True).to(device).eval()
    texts = [x['question'] + ' [SEP] ' + x['doc_text'] for x in flat]
    scores: list[float] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), args.crag_batch_size), desc='crag_evaluator'):
            encoded = tokenizer(texts[start:start + args.crag_batch_size], return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device)
            scores.extend(model(**encoded).logits[:, 0].float().cpu().tolist())
    return ([float(x) for x in scores], {'model': args.crag_model, 'device': str(device), 'batch_size': args.crag_batch_size})

def attach_scores(rows: list[dict[str, Any]], flat: list[dict[str, Any]], scores: list[float], extra: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    cursor = 0
    for (qidx, row) in enumerate(rows):
        docs = row.get('retrieved_docs') or []
        if not docs and row.get('retrieved_doc') is not None:
            docs = [row['retrieved_doc']]
        scored_docs = []
        for (rank, doc) in enumerate(docs):
            rec = flat[cursor]
            assert rec['qidx'] == qidx and rec['rank'] == rank
            detail = {}
            if 'parsed' in extra:
                detail['parsed'] = extra['parsed'][cursor]
            if 'errors' in extra and extra['errors'][cursor]:
                detail['error'] = extra['errors'][cursor]
            scored_docs.append({'rank': rank, 'score': float(scores[cursor]), 'doc': doc, 'detail': detail})
            cursor += 1
        out.append({**row, 'scored_docs': scored_docs})
    return out

def main() -> None:
    args = parse_args()
    if args.output is None:
        stem = args.input.stem
        args.output = REBUTTAL_ROOT / 'results' / 'top10_context' / 'scores' / f'{stem}_{args.selector}_scores.json'
    if args.output.exists() and (not args.overwrite):
        print(f'[skip completed] {args.output}')
        return
    (meta, rows) = load_topk(args.input, args.limit)
    flat = flatten_rows(rows)
    print(f'[score] selector={args.selector} questions={len(rows)} docs={len(flat)}')
    details: dict[str, Any] = {}
    if args.selector == 'dpr':
        scores = score_dpr(flat)
    elif args.selector == 'answer_oracle':
        scores = score_answer_oracle(flat)
    elif args.selector == 'csv':
        (scores, details) = score_csv(flat, args)
    elif args.selector == 'qwen36':
        (scores, details) = score_qwen36(flat, args)
    elif args.selector == 'qwen25':
        (scores, details) = score_qwen25(flat, args)
    elif args.selector == 'dus':
        (scores, details) = score_dus(flat, args)
    elif args.selector == 'cross_encoder':
        (scores, details) = score_cross_encoder(flat, args)
    elif args.selector == 'crag':
        (scores, details) = score_crag(flat, args)
    else:
        raise ValueError(args.selector)
    scored_rows = attach_scores(rows, flat, scores, details)
    score_values = np.asarray(scores, dtype=float)
    payload = {'meta': {'input': str(args.input), 'dataset': args.dataset or meta.get('dataset'), 'selector': args.selector, 'num_questions': len(rows), 'num_docs': len(flat), 'score_mean': float(score_values.mean()) if len(score_values) else None, 'score_min': float(score_values.min()) if len(score_values) else None, 'score_max': float(score_values.max()) if len(score_values) else None, 'details': {k: v for (k, v) in details.items() if k not in {'parsed', 'errors'}}}, 'data': scored_rows}
    dump_json(args.output, payload)
    print(f'[saved] {args.output}')
if __name__ == '__main__':
    main()
