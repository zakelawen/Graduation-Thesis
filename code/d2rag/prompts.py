from typing import Any, Dict, List
EVIDENCE_SEP = '\n\n---\n\n'
EVIDENCE_PREFIX = 'The following retrieved documents are provided verbatim before the question. Answer the upcoming multiple-choice question with only the final option letter (e.g., A, B, C, D, or E) without explanations.'
MAIN_SYSTEM = 'You are a helpful assistant. Your answer must be only the letter of the correct option (e.g., A, B, C, D, or E).'
STAGE1_SYSTEM = 'You are an expert at multiple-choice questions. According to the following question and options, respond with only the single, correct option, including its letter and full text.'

def build_stem(item: Dict[str, Any]) -> str:
    stem = (item.get('raw_question') or '').strip() or (item.get('question') or '').strip()
    if not stem:
        stem = 'Answer the multiple-choice question.'
    if not stem.lower().endswith('answer:'):
        stem += '\nAnswer:'
    return stem

def _evidence_text(docs: List[Dict[str, Any]], polarity: str) -> str:
    if not docs:
        return 'No retrieved documents were provided.' if polarity == 'pos' else 'No negative documents were provided.'
    tag = '' if polarity == 'pos' else 'NEG-'
    chunks = [f"[{tag}{i + 1}] Title: {d.get('title') or f'doc_{i}'}\n{d.get('text') or ''}" for (i, d) in enumerate(docs)]
    return EVIDENCE_PREFIX + '\n\n' + EVIDENCE_SEP.join(chunks)

def main_prompt(tokenizer, stem: str) -> str:
    messages = [{'role': 'system', 'content': MAIN_SYSTEM}, {'role': 'user', 'content': stem}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def ctx_prompt(tokenizer, stem: str, docs: List[Dict[str, Any]], polarity: str):
    evidence = _evidence_text(docs, polarity)
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'assistant', 'content': evidence}, {'role': 'user', 'content': stem}]
    return (tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), evidence[:1200])
