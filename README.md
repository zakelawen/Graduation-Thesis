# D²-RAG

## 主实验结果

| Method | MedMCQA Qwen2.5-7B | MedMCQA Llama3.1-8B | MedQA Qwen2.5-7B | MedQA Llama3.1-8B | MedExQA Qwen2.5-7B | MedExQA Llama3.1-8B | MMLU Qwen2.5-7B | MMLU Llama3.1-8B |
|---|---|---|---|---|---|---|---|---|
| PMC-Llama | 27.04 | | 16.94 | | 36.60 | | 37.01 | |
| MedAlpaca | 36.22 | | 50.04 | | 50.11 | | 60.85 | |
| Self-RAG | 38.32 | | 28.26 | | 71.38 | | 64.27 | |
| Llama3.1-70B | 78.28 | | 57.54 | | 77.34 | | 78.05 | |
| Zero Shot | 45.60 | 72.88 | 24.86 | 37.10 | 67.45 | 71.38 | 63.54 | 68.29 |
| CoT | 50.76 | 54.82 | 29.68 | 60.06 | 63.09 | 56.28 | 57.56 | 55.55 |
| Naive RAG | 54.04 | 62.50 | 40.10 | 46.16 | 69.36 | 66.28 | 63.96 | 61.77 |
| Adaptive RAG | 62.00 | 72.62 | 49.68 | 62.08 | 71.06 | 69.68 | 67.44 | 67.13 |
| DoLa | 67.16 | 71.30 | 56.72 | 63.64 | 73.19 | 70.85 | 71.83 | 70.49 |
| CAD | 62.54 | 63.48 | 55.04 | 57.12 | 68.94 | 62.66 | 67.07 | 62.93 |
| Ours | 67.16 | 70.44 | 57.38 | 57.12 | 74.68 | 70.74 | 73.84 | 70.37 |

## 消融实验结果

| Method | MedMCQA Llama3.1-8B | MedMCQA Qwen2.5-7B | MedQA Llama3.1-8B | MedQA Qwen2.5-7B |
|---|---|---|---|---|
| Ours | 70.44 | 67.16 | 57.12 | 57.38 |
| w/o Stage 1 & 2 | 62.50 | 54.04 | 46.16 | 40.10 |
| w/o Stage 1 | 69.44 | 67.14 | 57.62 | 56.96 |
| w/o Stage 2 | 66.16 | 59.02 | 51.46 | 41.38 |

## 结果文件

`results/index.csv` 列出表中每个数字对应的数据文件（`data_file`）和结果来源（`result`），以及正确题数 / 总题数。

```
results/
  index.csv
  main_table/<数据集>/<方法>__<模型>.jsonl.gz
  ablation/<设置>__<数据集>__<模型>.jsonl.gz
  notebook_outputs/
```

- `result` 为 `<文件> (<字段>)`：准确率由该文件逐题判定字段计算。
- `result` 为 `<notebook> cell <k>`：准确率为该 notebook 第 k 格的输出，对应的模型输出在 `data_file`。
- Self-RAG 的 MedMCQA / MedQA 两格为原始记录，没有数据文件。

核对全部结果：

```bash
python verify.py
```

数据文件为 gzip 压缩的 jsonl，用 `gzip -dk <文件>` 解压查看。

## 代码

```
code/
  d2rag/      对比解码、prompt、路由、SePer、判定
  scripts/    stage1_truth_values.py  seper_scores.py  route.py  decode.py  eval_outputs.py  summarize.py
  configs/settings.json
src/TruthTorchLM/
run.sh
```

环境：Python 3.11，`transformers==4.44.2`，`torch==2.5.1`，`accelerate==1.10.1`，`scikit-learn==1.5.1`，`bitsandbytes==0.50.2`。判定使用 gpt-4o-mini，运行前设置 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。

```bash
DATA=/path/to/data bash run.sh
```
