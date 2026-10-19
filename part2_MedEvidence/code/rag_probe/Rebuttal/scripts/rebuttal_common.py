from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
REBUTTAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def extract_doc_text(doc: Any) -> str:
    if isinstance(doc, dict):
        title = str(doc.get('title', '')).strip()
        text = str(doc.get('text', '')).strip()
        if title:
            return f'Title: {title}\nText: {text}'
        return text
    return str(doc).strip()

def as_answer_list(answers: Any) -> list[str]:
    if answers is None:
        return []
    if isinstance(answers, str):
        return [answers]
    return [str(a) for a in answers]

def load_retrieved_data(split: str) -> list[dict[str, Any]]:
    data_dir = Path(os.environ.get('REBUTTAL_RETRIEVED_DATA_DIR', PROJECT_ROOT / 'data' / 'final_retrieved'))
    path = data_dir / f'{split}.json'
    with path.open() as f:
        items = json.load(f)
    records = []
    for (idx, it) in enumerate(items):
        records.append({'id': it.get('id', idx), 'question': it['question'], 'doc_text': extract_doc_text(it['doc']), 'answers': as_answer_list(it.get('answers')), 'label': int(it['label']), 'source': str(it.get('source_dataset', 'unknown')).lower().strip()})
    return records

def load_gold_distractor_data(split: str) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / 'data' / 'final' / f'{split}.json'
    with path.open() as f:
        items = json.load(f)
    records = []
    for (idx, it) in enumerate(items):
        q = it['question']
        source = str(it.get('source_dataset', 'unknown')).lower().strip()
        answers = as_answer_list(it.get('answers'))
        records.append({'id': f"{it.get('id', idx)}:gold", 'question': q, 'doc_text': extract_doc_text(it['relevant_doc']), 'answers': answers, 'label': 1, 'source': source})
        records.append({'id': f"{it.get('id', idx)}:distractor", 'question': q, 'doc_text': extract_doc_text(it['distracting_doc']), 'answers': answers, 'label': 0, 'source': source})
    return records

def load_data(split: str, mode: str, limit: int | None=None) -> list[dict[str, Any]]:
    if mode == 'retrieved':
        records = load_retrieved_data(split)
    elif mode == 'gold_distractor':
        records = load_gold_distractor_data(split)
    else:
        raise ValueError(f'unknown mode: {mode}')
    return records if limit is None else records[:limit]

def build_rag_prompt(question: str, doc_text: str | None) -> str:
    if doc_text is None:
        return f'Question: {question}\nAnswer:'
    return f'Document: {doc_text}\n\nQuestion: {question}\nAnswer:'

def predict_from_scores(scores: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    if direction == 'greater':
        return (scores > threshold).astype(int)
    if direction == 'less':
        return (scores < threshold).astype(int)
    raise ValueError(direction)

def orient_scores(scores: np.ndarray, direction: str) -> np.ndarray:
    return scores if direction == 'greater' else -scores

def find_best_threshold(scores: list[float], labels: list[int], n_steps: int=400) -> tuple[float, float, str]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    (lo, hi) = (float(s.min() - 1e-06), float(s.max() + 1e-06))
    grid = np.linspace(lo, hi, n_steps)
    (best_acc, best_t, best_dir) = (-1.0, lo, 'greater')
    for t in grid:
        for direction in ('greater', 'less'):
            pred = predict_from_scores(s, float(t), direction)
            acc = float(accuracy_score(y, pred))
            if acc > best_acc:
                (best_acc, best_t, best_dir) = (acc, float(t), direction)
    return (best_t, best_acc, best_dir)

def evaluate(scores: list[float], labels: list[int], sources: list[str], threshold: float, direction: str) -> dict[str, dict[str, float | int]]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    src = np.asarray(sources)
    out: dict[str, dict[str, float | int]] = {}
    for subset in sorted(set(src.tolist())) + ['combined']:
        mask = np.ones(len(y), dtype=bool) if subset == 'combined' else src == subset
        (ss, yy) = (s[mask], y[mask])
        auroc = float('nan') if len(set(yy.tolist())) < 2 else float(roc_auc_score(yy, orient_scores(ss, direction)))
        out[subset] = {'auroc': auroc, 'accuracy': float(accuracy_score(yy, predict_from_scores(ss, threshold, direction))), 'avg_margin': float(np.abs(ss - threshold).mean()), 'n_samples': int(mask.sum()), 'n_relevant': int(yy.sum()), 'n_distracting': int((1 - yy).sum())}
    return out

def select_threshold_and_evaluate(train_scores: list[float], train_labels: list[int], eval_scores: list[float], eval_labels: list[int], eval_sources: list[str], train_sources: list[str] | None=None) -> dict[str, Any]:
    (threshold, train_best_acc, direction) = find_best_threshold(train_scores, train_labels)
    result: dict[str, Any] = {'threshold': threshold, 'direction': direction, 'train_best_acc': train_best_acc, 'eval': evaluate(eval_scores, eval_labels, eval_sources, threshold, direction)}
    if train_sources is not None:
        result['train'] = evaluate(train_scores, train_labels, train_sources, threshold, direction)
    return result

def print_eval(eval_results: dict[str, dict[str, Any]]) -> None:
    for (subset, m) in eval_results.items():
        print(f"  {subset:10s}: AUROC={m['auroc']:.4f}  Acc={m['accuracy']:.4f}  (n={m['n_samples']}, rel={m['n_relevant']}, dis={m['n_distracting']})")

def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(obj, f, indent=2)
