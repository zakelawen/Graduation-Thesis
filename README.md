# D²-RAG

## 1. 第一阶段分类器

| Methods | MedMCQA 需要检索 Recall | F1 | 不需要检索 Recall | F1 | MedQA 需要检索 Recall | F1 | 不需要检索 Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| FLARE (Llama3.1-8B) | 88.63 | 59.29 | 48.87 | 63.41 | 92.80 | 59.18 | 21.33 | 33.86 |
| Self-RAG-7B | 40.62 | 47.01 | 62.54 | 54.02 | 43.13 | 53.72 | 50.19 | 32.12 |
| Self-RAG-13B | 12.67 | 20.76 | 87.16 | 56.84 | 2.54 | 4.90 | 97.79 | 45.78 |
| Ours (Llama3.1-8B) | 93.73 | 69.91 | 79.49 | 79.78 | 93.35 | 64.45 | 22.84 | 35.64 |
| Ours (Qwen2.5-7B) | 82.46 | 60.32 | 46.46 | 59.26 | 90.55 | 66.58 | 25.90 | 38.51 |

数据：`results/classifier/`，计算：`python code/scripts/classifier_eval.py`

## 2. 主实验

| Method | MedMCQA Qwen | MedMCQA Llama | MedQA Qwen | MedQA Llama | MedExQA Qwen | MedExQA Llama | MMLU Qwen | MMLU Llama |
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

数据：`results/main_table/`

## 3. 消融实验

| Method | MedMCQA Llama | MedMCQA Qwen | MedQA Llama | MedQA Qwen |
|---|---|---|---|---|
| Ours | 70.44 | 67.16 | 57.12 | 57.38 |
| w/o Stage 1 & 2 | 62.50 | 54.04 | 46.16 | 40.10 |
| w/o Stage 1 | 69.44 | 67.14 | 57.62 | 56.96 |
| w/o Stage 2 | 66.16 | 59.02 | 51.46 | 41.38 |

数据：`results/ablation/`

## 4. 超参数分析

| alpha_pos | alpha_neg | MedMCQA Qwen | MedMCQA Llama | MedQA Qwen | MedQA Llama | MedExQA Qwen | MedExQA Llama | MMLU Qwen | MMLU Llama |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.05 | 67.12 | 70.10 | 57.38 | 56.42 | 74.36 | 70.11 | 73.60 | 69.70 |
| 0.1 | 0.1 | 66.66 | 70.28 | 57.30 | 56.58 | 74.26 | 70.00 | 73.60 | 69.63 |
| 0.1 | 0.2 | 67.16 | 70.38 | 57.26 | 56.44 | 74.26 | 70.00 | 73.60 | 69.57 |
| 0.5 | 0.05 | 67.00 | 70.22 | 57.32 | 56.36 | 74.57 | 70.64 | 73.84 | 69.88 |
| 0.5 | 0.1 | 67.08 | 70.40 | 57.24 | 56.64 | 74.47 | 70.64 | 73.84 | 69.82 |
| 0.5 | 0.2 | 67.16 | 70.40 | 57.30 | 56.62 | 74.68 | 70.53 | 73.84 | 69.76 |
| 1.0 | 0.05 | 66.98 | 70.36 | 57.28 | 56.96 | 74.47 | 70.64 | 73.84 | 70.37 |
| 1.0 | 0.1 | 67.08 | 70.40 | 57.30 | 57.08 | 74.36 | 70.64 | 73.84 | 70.30 |
| 1.0 | 0.2 | 67.08 | 70.44 | 57.14 | 57.12 | 74.47 | 70.74 | 73.84 | 70.24 |

数据：`results/hyperparameter/`

## 结果文件

`results/index.csv` 列出每个数字对应的数据文件和结果来源。核对：

```bash
python verify.py
```

数据文件为 gzip 压缩，`gzip -dk <文件>` 解压查看。Self-RAG 主实验 MedMCQA / MedQA 两格为原始记录。

## 代码

```
code/d2rag/      对比解码、prompt、路由、SePer、判定
code/scripts/    stage1_truth_values.py  route.py  seper_scores.py  decode.py  eval_outputs.py  summarize.py  classifier_eval.py
code/configs/    settings.json
src/TruthTorchLM/
run.sh
```

Python 3.11，transformers 4.44.2，torch 2.5.1，accelerate 1.10.1，scikit-learn 1.5.1，bitsandbytes 0.50.2。运行前设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`。

```bash
DATA=/path/to/data bash run.sh
```
