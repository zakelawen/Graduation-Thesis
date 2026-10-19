#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export RAG_PROBE=${RAG_PROBE:-$HERE/code/rag_probe}
export MT_WORK=${MT_WORK:-$PWD}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-0}
M=$HERE/code/medical
PY=${PY:-python}
MODEL=${MODEL:-qwen3_4b}
BGE_MODEL=${BGE_MODEL:-BAAI/bge-reranker-large}
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
SELFRAG_MODEL=${SELFRAG_MODEL:-selfrag/selfrag_llama2_7b}
DUS_MODEL=${DUS_MODEL:-}
CE_MODEL=${CE_MODEL:-}
cd "$MT_WORK"
mkdir -p data labels scores generations manifests results results_medical
IN=$MT_WORK/data/mirage_medcpt_top10.json
S=$MT_WORK/scores
score() { (cd "$RAG_PROBE" && $PY Rebuttal/scripts/top10_score_contexts.py --input "$IN" --dataset mirage "$@"); }
train() {
  (cd "$RAG_PROBE" && $PY "$M/train_csv_lowmem.py" --model "$1" --str_layers "$2" --cls_layers "$3" --lam 5.0 --cos_temp 0.1 \
    --lr "$4" --ema_decay "$5" --batch_size "$6" --num_epochs 20 --patience 5 --resume \
    --data_dir "$MT_WORK/data/medical_retrieved" --results_dir "$7")
}
search() {
  R=$MT_WORK/results_medical
  train "$1" "$3" "$4" "$7" "$8" "$9" "$R/stageA"
  if $PY "$M/layer_search.py" --run-name "$2" --dirs "$R/stageA" --inject "$3" --cls "$4" | grep -q "^BOUNDARY"; then
    train "$1" "$5" "$6" "$7" "$8" "$9" "$R/stageB"
  fi
  $PY "$M/layer_search.py" --run-name "$2" --dirs "$R/stageA" "$R/stageB" --merge-to "$R/csv_retrieved"
}

case "${1:-}" in
  corpus)
    $PY "$M/build_statpearls.py" --tar "$STATPEARLS_TAR" --out-dir corpus/statpearls/chunk
    for c in pubmed textbooks statpearls; do $PY "$M/encode_corpus.py" --corpus-dir corpus/$c/chunk --out-dir emb/$c; done
    ;;
  retrieve)
    $PY "$M/retrieve_mirage.py" --benchmark data/mirage/benchmark.json --emb-root emb --corpus-root corpus --out "$IN" --query-max-length 512
    ;;
  train_questions)
    $PY "$M/sample_train_questions.py"
    $PY "$M/extend_train_questions.py"
    $PY "$M/retrieve_mirage.py" --benchmark data/train_questions.json --emb-root emb --corpus-root corpus --out data/train_medcpt_top10.json --query-max-length 512
    $PY "$M/extend_train_questions_2.py"
    $PY "$M/retrieve_mirage.py" --benchmark data/train_questions_extra.json --emb-root emb --corpus-root corpus --out data/train_extra_medcpt_top10.json --query-max-length 512
    $PY - <<'EOF'
import json
a = json.load(open("data/train_medcpt_top10.json")); b = json.load(open("data/train_extra_medcpt_top10.json"))
have = {(r["dataset"], r["id"]) for r in a["data"]}
a["data"] += [r for r in b["data"] if (r["dataset"], r["id"]) not in have]
json.dump(a, open("data/train_medcpt_top10.json", "w"), ensure_ascii=False)
EOF
    ;;
  label)
    $PY "$M/label_kakou.py" --retrieval "$IN" --ids "$HERE/results/classification/test_sample_ids.json" --output labels/mirage_top1_claude.jsonl --workers 8
    $PY "$M/label_kakou.py" --retrieval data/train_medcpt_top10.json --ids labels/train_medmcqa_ids.json --output labels/train_top1_claude.jsonl --workers 8
    ;;
  train_data)
    $PY "$M/build_medical_train_data.py"
    ;;
  score)
    score --selector dpr --output "$S/retriever_score.json"
    $PY "$M/score_utility_ce_prob.py" --input "$IN" --model "$CE_MODEL" --output "$S/utility_cross_encoder.json"
    score --selector cross_encoder --cross-encoder-model "$BGE_MODEL" --cross-encoder-batch-size 64 --output "$S/bge_reranker.json"
    score --selector crag --crag-batch-size 64 --output "$S/crag_evaluator.json"
    score --selector dus --dus-selector-path "$DUS_MODEL" --dus-device cuda:0 --dus-batch-size 16 --dus-num-samples 1 --dus-temperature 0 --output "$S/dus_utilityqwen.json"
    score --selector qwen25 --qwen25-model "$JUDGE_MODEL" --qwen25-device cuda:0 --qwen25-batch-size 4 --output "$S/llm_utility_judge.json"
    (cd "$RAG_PROBE" && $PY Rebuttal/scripts/selfrag_relevance_top10.py --input "$IN" --output "$S/selfrag7b_relevance.json" --model "$SELFRAG_MODEL" \
      --selector-name selfrag7b --tensor-parallel-size 1 --batch-size 128 --chunk-size 256 --gpu-memory-utilization 0.90 --max-model-len 4096 --cache-dir "$MT_WORK/cache/selfrag7b")
    $PY "$M/make_random_scores.py"
    ;;
  score_ours)
    score --selector csv --csv-model "$MODEL" --csv-batch-size 8 --output "$S/ours_$MODEL.json"
    ;;
  train)
    if [[ "$MODEL" == qwen3_4b ]]; then
      search qwen3_4b qwen3_4b_retrieved 17,19,21 20,22,24 3,7,11,15,19,23,27,31 7,11,15,19,23,27,31,35 1e-3 0.999 4
    else
      search gemma2b gemma2b_retrieved 12,14,16 16,18,20 2,5,8,11,14,17,20,23 5,8,11,14,17,20,23,25 2e-3 0.99 8
    fi
    ;;
  score_ours_med)
    CSV_RETRIEVED_DIR=$MT_WORK/results_medical/csv_retrieved score --selector csv --csv-model "$MODEL" --csv-batch-size 8 --skip-csv-validation --output "$S/ours_med_$MODEL.json"
    ;;
  generate)
    $PY "$M/build_medical_manifest.py" --work "$MT_WORK" --model "$MODEL" --output manifests/manifest_$MODEL.json
    (cd "$RAG_PROBE" && $PY Rebuttal/scripts/top10_select_generate_vllm.py --manifest "$MT_WORK/manifests/manifest_$MODEL.json" --model "$MODEL" \
      --output-dir "$MT_WORK/generations" --tensor-parallel-size 1 --gpu-memory-utilization 0.90 --max-num-seqs 64 --max-num-batched-tokens 8192 --dtype bfloat16)
    ;;
  evaluate)
    $PY "$M/eval_mcq.py" --retrieval "$IN" --glob "generations/${MODEL}_mirage_*.json" --out results/downstream_$MODEL.tsv
    $PY "$M/eval_classification.py" --labels labels/mirage_top1_claude.jsonl --scores-dir scores --out results/classification.tsv
    ;;
  checks)
    $PY "$M/extract_invalid_llm.py"
    $PY "$M/judge_audit.py"
    ;;
  *)
    echo "usage: [MODEL=qwen3_4b|gemma2b] $0 {corpus|retrieve|train_questions|label|train_data|score|score_ours|train|score_ours_med|generate|evaluate|checks}" >&2
    exit 2
    ;;
esac
