import os
import argparse
import json
from pathlib import Path
RAG_PROBE = os.environ.get('RAG_PROBE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rag_probe'))
MT_WORK = os.environ.get('MT_WORK', os.getcwd())
RAG = Path(RAG_PROBE)
CLASS = RAG / 'Rebuttal/results/popqa1106_generalization/classification'
SELFRAG7B = RAG / 'Rebuttal/results/selfrag7b_classification/retrieved.json'

def frozen(path):
    obj = json.loads(path.read_text(encoding='utf-8'))
    return (float(obj['threshold']), str(obj.get('direction', 'greater')))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, default=Path(MT_WORK))
    ap.add_argument('--model', required=True, choices=('qwen3_4b', 'gemma9b', 'gemma2b'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    scores = args.work / 'scores'
    raw = args.work / 'data/mirage_medcpt_top10.json'
    jobs = [{'input': str(raw), 'dataset': 'mirage', 'selector_name': 'no_retrieval', 'rule': 'none', 'decoding_method': 'no_doc'}, {'input': str(scores / 'retriever_score.json'), 'dataset': 'mirage', 'selector_name': 'medcpt_top1', 'rule': 'top1', 'decoding_method': 'vanilla_rag'}, {'input': str(scores / 'retriever_score.json'), 'dataset': 'mirage', 'selector_name': 'medcpt_top10', 'rule': 'top10', 'decoding_method': 'vanilla_rag'}, {'input': str(scores / 'retriever_score.json'), 'dataset': 'mirage', 'selector_name': 'medcpt_top3', 'rule': 'score_topk', 'select_top_k': 3, 'score_direction': 'greater', 'decoding_method': 'vanilla_rag'}, {'input': str(scores / 'random_seed0.json'), 'dataset': 'mirage', 'selector_name': 'random_top3', 'rule': 'score_topk', 'select_top_k': 3, 'score_direction': 'greater', 'decoding_method': 'vanilla_rag'}, {'input': str(scores / 'bge_reranker.json'), 'dataset': 'mirage', 'selector_name': 'bge_top3', 'rule': 'score_topk', 'select_top_k': 3, 'score_direction': 'greater', 'decoding_method': 'vanilla_rag'}]
    specs = []
    for (name, threshold_file, score_file) in specs:
        if not score_file.exists():
            print(f'[skip] {name}: no scores yet')
            continue
        (threshold, direction) = frozen(threshold_file)
        jobs.append({'input': str(score_file), 'dataset': 'mirage', 'selector_name': name, 'rule': 'score_threshold', 'threshold': threshold, 'score_direction': direction, 'decoding_method': 'vanilla_rag'})
    seper = scores / f'seper_{args.model}.json'
    if seper.exists():
        jobs.append({'input': str(seper), 'dataset': 'mirage', 'selector_name': 'seper_oracle', 'rule': 'score_threshold', 'threshold': 1e-12, 'score_direction': 'greater', 'decoding_method': 'vanilla_rag'})
    ours = scores / f'ours_{args.model}.json'
    if ours.exists():
        jobs.append({'input': str(ours), 'dataset': 'mirage', 'selector_name': 'ours', 'rule': 'score_threshold', 'threshold': 0.5, 'score_direction': 'greater', 'decoding_method': 'vanilla_rag'})
    ours_med = scores / f'ours_med_{args.model}.json'
    if ours_med.exists():
        jobs.append({'input': str(ours_med), 'dataset': 'mirage', 'selector_name': 'ours_medical', 'rule': 'score_threshold', 'threshold': 0.5, 'score_direction': 'greater', 'decoding_method': 'vanilla_rag'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'[saved] {args.output}; jobs={len(jobs)}')
if __name__ == '__main__':
    main()
