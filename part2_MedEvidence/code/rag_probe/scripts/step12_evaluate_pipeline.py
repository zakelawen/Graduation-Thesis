from __future__ import annotations
import argparse
import ast
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / 'results' / 'pipeline'

def load_json(path: Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def normalize_answer(s: Any) -> str:
    if s is None:
        return ''
    s = str(s)

    def lower(text: str) -> str:
        return text.lower()

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join((ch for ch in text if ch not in exclude))

    def remove_articles(text: str) -> str:
        return re.sub('\\b(a|an|the)\\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction: Any, ground_truth: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: Any, ground_truth: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def metric_max_over_ground_truths(metric_fn, prediction: Any, ground_truths: list[Any]) -> float:
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]
    if len(ground_truths) == 0:
        return 0.0
    return max((metric_fn(prediction, gt) for gt in ground_truths))

def _try_parse_string_collection(s: str) -> Any | None:
    t = s.strip()
    if not t:
        return None
    collection_like = t.startswith('[') and t.endswith(']') or (t.startswith('(') and t.endswith(')')) or (t.startswith('{') and t.endswith('}'))
    if not collection_like:
        return None
    try:
        return json.loads(t)
    except Exception:
        pass
    try:
        return ast.literal_eval(t)
    except Exception:
        return None

def _flatten_answers(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        out: list[Any] = []
        for v in x:
            out.extend(_flatten_answers(v))
        return out
    if isinstance(x, tuple):
        out: list[Any] = []
        for v in x:
            out.extend(_flatten_answers(v))
        return out
    if isinstance(x, dict):
        for key in ('text', 'answer', 'answers', 'aliases'):
            if key in x:
                return _flatten_answers(x[key])
        return [str(x)]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        parsed = _try_parse_string_collection(s)
        if parsed is not None:
            return _flatten_answers(parsed)
        return [s]
    return [str(x)]

def get_answers(item: dict[str, Any]) -> list[str]:
    raw_answers = item.get('answers', [])
    aliases = _flatten_answers(raw_answers)
    out: list[str] = []
    seen: set[str] = set()
    for a in aliases:
        a_str = '' if a is None else str(a).strip()
        if not a_str:
            continue
        if a_str not in seen:
            seen.add(a_str)
            out.append(a_str)
    return out

def get_prediction(item: dict[str, Any]) -> str:
    gen = item.get('generation', {})
    if not isinstance(gen, dict):
        return ''
    pred = gen.get('prediction', '')
    return '' if pred is None else str(pred)

def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))

def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {'n': 0, 'em': None, 'f1': None}
    return {'n': len(items), 'em': mean([x['em'] for x in items]), 'f1': mean([x['f1'] for x in items])}

def evaluate_pipeline_file(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict) or 'outputs' not in payload:
        raise ValueError(f'Not a Step11 generation output file: {path}')
    meta = payload.get('meta', {})
    if not isinstance(meta, dict):
        meta = {}
    outputs = payload.get('outputs', [])
    if not isinstance(outputs, list):
        raise ValueError(f'Bad outputs field in {path}')
    per_sample = []
    for (row_idx, item) in enumerate(outputs):
        if not isinstance(item, dict):
            continue
        pred = get_prediction(item)
        answers = get_answers(item)
        em = metric_max_over_ground_truths(exact_match_score, pred, answers)
        f1 = metric_max_over_ground_truths(f1_score, pred, answers)
        csv_info = item.get('csv', {}) if isinstance(item.get('csv', {}), dict) else {}
        gen_info = item.get('generation', {}) if isinstance(item.get('generation', {}), dict) else {}
        per_sample.append({'idx': item.get('idx', row_idx), 'question': item.get('question', ''), 'answers': answers, 'num_gold_aliases': len(answers), 'prediction': pred, 'em': float(em), 'f1': float(f1), 'source_dataset': item.get('source_dataset', meta.get('dataset')), 'p_relevant': csv_info.get('p_relevant'), 'trusted_by_tau': csv_info.get('trusted_by_tau'), 'actual_mode': gen_info.get('actual_mode'), 'alpha_used': gen_info.get('alpha_used')})
    overall = summarize_group(per_sample)
    by_trust = {}
    for val in [True, False]:
        group = [x for x in per_sample if x.get('trusted_by_tau') is val]
        by_trust[str(val).lower()] = summarize_group(group)
    by_mode = {}
    modes = sorted(set((str(x.get('actual_mode')) for x in per_sample)))
    for mode in modes:
        by_mode[mode] = summarize_group([x for x in per_sample if str(x.get('actual_mode')) == mode])
    p_vals = [x.get('p_relevant') for x in per_sample if isinstance(x.get('p_relevant'), (int, float))]
    p_vals_f = [float(x) for x in p_vals]
    p_diag = {'mean': mean(p_vals_f), 'min': float(min(p_vals_f)) if p_vals_f else None, 'max': float(max(p_vals_f)) if p_vals_f else None}
    answer_alias_counts = [x.get('num_gold_aliases', 0) for x in per_sample]
    alias_diag = {'mean_num_aliases': mean([float(x) for x in answer_alias_counts]), 'max_num_aliases': max(answer_alias_counts) if answer_alias_counts else None, 'num_multi_alias_samples': int(sum((1 for x in answer_alias_counts if x > 1)))}
    result = {'input_file': str(path), 'meta': meta, 'metrics': {'overall': overall, 'by_trusted_by_tau': by_trust, 'by_actual_mode': by_mode, 'p_relevant': p_diag, 'answer_aliases': alias_diag}, 'per_sample': per_sample}
    return result

def is_generation_output(path: Path) -> bool:
    name = path.name
    if name.startswith('eval_'):
        return False
    if 'validation' in name:
        return False
    if 'summary' in name:
        return False
    if not name.endswith('.json'):
        return False
    try:
        obj = load_json(path)
    except Exception:
        return False
    return isinstance(obj, dict) and isinstance(obj.get('outputs'), list)

def get_meta_for_filter(path: Path) -> dict[str, Any]:
    try:
        obj = load_json(path)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    meta = obj.get('meta', {})
    return meta if isinstance(meta, dict) else {}

def normalize_dataset_name(x: Any) -> str:
    s = str(x).lower().strip()
    aliases = {'tqa': 'triviaqa', 'trivia': 'triviaqa', 'trivia_qa': 'triviaqa', 'nq': 'nq', 'naturalquestions': 'nq', 'natural_questions': 'nq'}
    return aliases.get(s, s)

def matches_filters(path: Path, args) -> bool:
    meta = get_meta_for_filter(path)
    if args.model:
        allowed_models = {str(x) for x in args.model}
        if str(meta.get('model')) not in allowed_models:
            return False
    if args.dataset:
        allowed_datasets = {normalize_dataset_name(x) for x in args.dataset}
        if normalize_dataset_name(meta.get('dataset')) not in allowed_datasets:
            return False
    if args.method:
        allowed_methods = {str(x) for x in args.method}
        if str(meta.get('method')) not in allowed_methods:
            return False
    return True

def expand_glob_pattern(pattern: str) -> list[Path]:
    p = Path(pattern)
    if p.is_absolute():
        anchor = Path(p.anchor)
        rel = str(p.relative_to(anchor))
        return list(anchor.glob(rel))
    return list(PROJECT_ROOT.glob(pattern))

def collect_input_files(args) -> list[Path]:
    files: list[Path] = []
    if args.input:
        for x in args.input:
            files.append(Path(x))
    if args.glob:
        for pat in args.glob:
            files.extend(expand_glob_pattern(pat))
    scan_mode = args.all or (not args.input and (not args.glob) and (args.model or args.dataset or args.method))
    if scan_mode:
        files.extend(sorted(PIPELINE_DIR.glob('*.json')))
    out: list[Path] = []
    seen: set[str] = set()
    for p in files:
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    if scan_mode:
        out = [p for p in out if p.exists() and is_generation_output(p)]
    else:
        missing = [p for p in out if not p.exists()]
        if missing:
            raise FileNotFoundError('Missing input files:\n' + '\n'.join((str(x) for x in missing)))
        out = [p for p in out if is_generation_output(p)]
    if args.model or args.dataset or args.method:
        out = [p for p in out if matches_filters(p, args)]
    return sorted(out)

def summary_row(eval_result: dict[str, Any], eval_path: Path) -> dict[str, Any]:
    meta = eval_result.get('meta', {})
    overall = eval_result['metrics']['overall']
    p_rel = eval_result['metrics'].get('p_relevant', {})
    alias_info = eval_result['metrics'].get('answer_aliases', {})
    return {'eval_file': str(eval_path), 'input_file': eval_result.get('input_file'), 'model': meta.get('model'), 'dataset': meta.get('dataset'), 'method': meta.get('method'), 'alpha': meta.get('alpha'), 'tau': meta.get('tau'), 'contrast_layer': meta.get('contrast_layer'), 'temperature': meta.get('temperature'), 'top_p': meta.get('top_p'), 'max_new_tokens': meta.get('max_new_tokens'), 'n': overall.get('n'), 'em': overall.get('em'), 'f1': overall.get('f1'), 'trusted_docs': meta.get('trusted_docs'), 'trusted_rate': meta.get('trusted_rate'), 'p_relevant_mean': p_rel.get('mean'), 'csv_best_key': meta.get('csv_best_key'), 'mean_num_gold_aliases': alias_info.get('mean_num_aliases'), 'max_num_gold_aliases': alias_info.get('max_num_aliases'), 'num_multi_alias_samples': alias_info.get('num_multi_alias_samples')}

def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def write_global_summary_from_eval_files(pipeline_dir: Path=PIPELINE_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(pipeline_dir.glob('eval_*.json')):
        if 'summary' in path.name or 'validation' in path.name:
            continue
        try:
            obj = load_json(path)
        except Exception as exc:
            rows.append({'eval_file': str(path), 'note': f'load_error: {exc!r}'})
            continue
        if not isinstance(obj, dict) or 'metrics' not in obj:
            continue
        rows.append(summary_row(obj, path))

    def sort_key(row: dict[str, Any]) -> tuple[str, str, str, float, str]:
        try:
            alpha = float(row.get('alpha') if row.get('alpha') is not None else -1.0)
        except Exception:
            alpha = -1.0
        return (str(row.get('model') or ''), str(row.get('dataset') or ''), str(row.get('method') or ''), alpha, str(row.get('eval_file') or ''))
    rows.sort(key=sort_key)
    summary_json = pipeline_dir / 'pipeline_eval_summary.json'
    summary_csv = pipeline_dir / 'pipeline_eval_summary.csv'
    save_json(rows, summary_json)
    write_summary_csv(rows, summary_csv)
    print(f'Wrote Step12 global summary from {len(rows)} eval files')
    print(f'Summary JSON → {summary_json}')
    print(f'Summary CSV  → {summary_csv}')
    return rows

def run_answer_parser_self_test() -> None:
    examples = [({'answers': "['Scorpio', 'Skorpio', 'Scorpio (disambiguation)']"}, ['Scorpio', 'Skorpio', 'Scorpio (disambiguation)']), ({'answers': '["A", "B"]'}, ['A', 'B']), ({'answers': ['A', 'B']}, ['A', 'B']), ({'answers': [['A', 'B'], 'C']}, ['A', 'B', 'C']), ({'answers': 'May'}, ['May'])]
    for (item, expected) in examples:
        got = get_answers(item)
        assert got == expected, f'get_answers({item!r}) = {got!r}, expected {expected!r}'
    em = metric_max_over_ground_truths(exact_match_score, 'Scorpio', get_answers({'answers': "['Scorpio', 'Skorpio', 'Scorpio (disambiguation)']"}))
    assert em == 1.0, f'Expected EM=1.0 for Scorpio alias test, got {em}'
    print('Answer parser self-test passed.')

def main() -> None:
    parser = argparse.ArgumentParser(description='Step12: Evaluate Step11 pipeline generations')
    parser.add_argument('--input', nargs='*', default=None, help='One or more Step11 generation json files.')
    parser.add_argument('--glob', nargs='*', default=None, help="Glob pattern(s), e.g. 'results/pipeline/gemma2b_nq_*.json'.")
    parser.add_argument('--all', action='store_true', help='Evaluate all Step11 generation outputs in results/pipeline/.')
    parser.add_argument('--model', nargs='+', default=None, help='Evaluate only outputs whose meta.model matches, e.g. --model gemma2b')
    parser.add_argument('--dataset', nargs='+', default=None, help="Evaluate only outputs whose meta.dataset matches, e.g. --dataset nq triviaqa. 'tqa' is accepted.")
    parser.add_argument('--method', nargs='+', default=None, help='Evaluate only outputs whose meta.method matches, e.g. --method csv_gated_cad')
    parser.add_argument('--self_test', action='store_true', help='Run answer parser self-test and exit.')
    args = parser.parse_args()
    if args.self_test:
        run_answer_parser_self_test()
        return
    if not args.input and (not args.glob) and (not args.all) and (not (args.model or args.dataset or args.method)):
        parser.error('Provide --input, --glob, --all, or at least one filter such as --model gemma2b')
    files = collect_input_files(args)
    if not files:
        raise RuntimeError('No valid Step11 generation output files found.')
    print('\n' + '=' * 80)
    print('Step 12: Pipeline EM / F1 evaluation')
    print('=' * 80)
    print(f'Project root: {PROJECT_ROOT}')
    print(f'Pipeline dir: {PIPELINE_DIR}')
    if args.model:
        print(f'Model filter: {args.model}')
    if args.dataset:
        print(f'Dataset filter: {args.dataset}')
    if args.method:
        print(f'Method filter: {args.method}')
    print(f'Files to evaluate: {len(files)}')
    for p in files:
        try:
            print(f'  - {p.relative_to(PROJECT_ROOT)}')
        except ValueError:
            print(f'  - {p}')
    summary_rows = []
    for path in files:
        print(f'\n=== Evaluating {path} ===')
        result = evaluate_pipeline_file(path)
        out_path = PIPELINE_DIR / f'eval_{path.stem}.json'
        save_json(result, out_path)
        overall = result['metrics']['overall']
        print(f"  N={overall['n']}  EM={overall['em']:.4f}  F1={overall['f1']:.4f}")
        alias_info = result['metrics'].get('answer_aliases', {})
        print(f"  Answer aliases: mean={alias_info.get('mean_num_aliases'):.2f}  max={alias_info.get('max_num_aliases')}  multi_alias_samples={alias_info.get('num_multi_alias_samples')}")
        print(f'  Saved → {out_path}')
        summary_rows.append(summary_row(result, out_path))
    write_global_summary_from_eval_files(PIPELINE_DIR)
    print('\n' + '=' * 80)
    print('Step12 done.')
    print(f"Full summary JSON → {PIPELINE_DIR / 'pipeline_eval_summary.json'}")
    print(f"Full summary CSV  → {PIPELINE_DIR / 'pipeline_eval_summary.csv'}")
    print('=' * 80)
if __name__ == '__main__':
    main()
