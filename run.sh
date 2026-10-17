#!/bin/bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PWD/code:$PYTHONPATH"
DATA=${DATA:-data}
for ds in medmcqa medqa medexqa mmlu; do
  for m in llama qwen; do
    out=runs/${ds}_${m}
    python code/scripts/stage1_truth_values.py --model $m --calib $DATA/$ds/dev.jsonl --test $DATA/$ds/test.jsonl --out $out/stage1.jsonl
    python code/scripts/seper_scores.py --model $m --input $DATA/$ds/test_with_docs.jsonl --out $out/seper.jsonl
    python code/scripts/route.py --model $m --stage1 $out/stage1.jsonl --seper $out/seper.jsonl --out $out/routed.jsonl --data-root $DATA
    python code/scripts/decode.py --model $m --dataset $ds --routed $out/routed.jsonl --out-dir $out/decode
    python code/scripts/eval_outputs.py $out/decode/*.jsonl
  done
done
python code/scripts/summarize.py runs/*/decode/*.jsonl
