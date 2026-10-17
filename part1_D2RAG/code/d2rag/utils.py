import json
import os
from typing import Any, Dict, Iterable, List

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding='utf-8') as f:
        text = f.read().strip()
    if text.startswith('['):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()

def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_serializable(v) for (k, v) in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    return obj

def question_key(text: str) -> str:
    return (text or '').strip()
