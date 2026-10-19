# Part 2: 医疗问答中的证据判断

检索语料 MedCorp（PubMed + StatPearls + 教科书），检索器 MedCPT，测试集 MIRAGE（MedMCQA / MedQA / MMLU-Med / BioASQ / PubMedQA，共 7,663 题）。
Ours：零样本为通用域训练的 steering vector 与原型直接迁移；医疗为在医疗训练集上重新训练。

## 1. 分类实验

MIRAGE 抽样 2,000 题的 top-1 段落，Claude Opus 5.5 标注，去掉 neutral 后 1,200 题（relevant 541 / distracting 659）。阈值：基线沿用通用域冻结阈值，Ours 为 0.5；MedCPT score 没有阈值。

| Method | AUROC | AUPRC | Acc | Macro-F1 |
|---|---|---|---|---|
| Ours Qwen3-4B (medical) | 87.82 | 84.15 | 80.58 | 80.55 |
| Ours Gemma2-2B (medical) | 85.37 | 80.08 | 78.92 | 78.73 |
| Ours Qwen3-4B (zero-shot) | 83.54 | 82.68 | 77.50 | 76.95 |
| Ours Gemma2-2B (zero-shot) | 71.32 | 62.39 | 58.50 | 55.24 |
| LLM Utility Judge (Qwen2.5-7B) | 85.90 | 82.48 | 77.83 | 77.08 |
| Self-RAG-7B | 80.78 | 76.37 | 72.75 | 72.74 |
| BGE Reranker | 78.73 | 76.75 | 71.75 | 69.88 |
| Utility CE | 78.26 | 74.23 | 63.50 | 62.50 |
| MedCPT score | 69.22 | 64.09 |  |  |
| CRAG Evaluator | 56.23 | 57.49 | 60.33 | 49.19 |
| DUS | 50.27 | 45.22 | 45.42 | 32.00 |

各子集 AUROC：

| Method | MedMCQA | MedQA | MMLU-Med | BioASQ | PubMedQA |
|---|---|---|---|---|---|
| Ours Qwen3-4B (medical) | 85.69 | 79.75 | 83.71 | 69.52 | 67.37 |
| Ours Gemma2-2B (medical) | 83.26 | 78.04 | 81.35 | 68.16 | 70.61 |
| Ours Qwen3-4B (zero-shot) | 78.00 | 68.24 | 76.96 | 75.50 | 72.68 |
| Ours Gemma2-2B (zero-shot) | 71.38 | 73.70 | 73.85 | 64.17 | 43.89 |
| LLM Utility Judge (Qwen2.5-7B) | 80.63 | 76.41 | 82.89 | 59.67 | 76.04 |
| Self-RAG-7B | 73.90 | 69.74 | 74.24 | 60.18 | 67.70 |
| BGE Reranker | 71.08 | 55.38 | 75.23 | 52.41 | 80.91 |
| Utility CE | 69.51 | 62.15 | 70.88 | 55.08 | 70.20 |
| MedCPT score | 58.65 | 50.53 | 66.02 | 64.71 | 64.17 |
| CRAG Evaluator | 51.15 | 48.34 | 49.12 | 48.76 | 81.31 |
| DUS | 50.42 | 50.00 | 49.83 | 50.00 | 50.00 |

数据：`results/classification/`

## 2. 下游问答

MIRAGE 全部 7,663 题，选项准确率（%），规则提取选项字母判分。Macro 为五个子集的简单平均，Micro 按题数加权；#Docs 为平均提供的段落数。

**Qwen3-4B**

| Method | #Docs | MedMCQA | MedQA | MMLU-Med | BioASQ | PubMedQA | Macro | Micro |
|---|---|---|---|---|---|---|---|---|
| No retrieval | 0.0 | 58.26 | 60.80 | 77.13 | 74.76 | 56.60 | 65.51 | 62.59 |
| MedCPT top-1 | 1.0 | 56.49 | 61.12 | 75.76 | 78.64 | **57.80** | 65.96 | 61.87 |
| Random-3 | 3.0 | 57.16 | 60.09 | 74.56 | 77.83 | 50.20 | 63.97 | 61.33 |
| MedCPT top-3 | 3.0 | 58.09 | 60.80 | 75.94 | 79.45 | 56.40 | 66.14 | 62.69 |
| BGE rerank top-3 | 3.0 | 58.83 | 60.33 | 74.93 | **81.07** | 57.20 | 66.47 | 63.06 |
| MedCPT top-10 | 10.0 | 60.41 | 60.72 | 76.31 | 80.26 | 57.20 | 66.98 | 64.11 |
| Ours (zero-shot) | 2.0 | 60.22 | **61.90** | 76.95 | 79.13 | 57.40 | 67.12 | 64.22 |
| Ours (medical) | 2.9 | **60.46** | 61.67 | **77.23** | 80.26 | 57.20 | **67.36** | **64.43** |

**Gemma2-2B**

| Method | #Docs | MedMCQA | MedQA | MMLU-Med | BioASQ | PubMedQA | Macro | Micro |
|---|---|---|---|---|---|---|---|---|
| No retrieval | 0.0 | 40.00 | 35.11 | 53.81 | **62.30** | 33.20 | 44.88 | 42.50 |
| MedCPT top-1 | 1.0 | 39.28 | 35.19 | 52.25 | 55.99 | 34.80 | 43.50 | 41.50 |
| Random-3 | 3.0 | 40.81 | 34.72 | 51.79 | 50.16 | 32.80 | 42.06 | 41.59 |
| MedCPT top-3 | 3.0 | 41.57 | 35.51 | 52.89 | 57.44 | 37.60 | 45.00 | 43.19 |
| BGE rerank top-3 | 3.0 | 42.53 | 37.31 | 53.63 | 55.50 | **39.40** | **45.67** | 44.08 |
| MedCPT top-10 | 10.0 | 43.08 | 35.43 | 51.97 | 49.19 | 37.60 | 43.45 | 43.21 |
| Ours (zero-shot) | 7.6 | 43.06 | 33.86 | 53.08 | 48.71 | 38.00 | 43.34 | 43.08 |
| Ours (medical) | 2.3 | **44.82** | **38.02** | **56.38** | 50.81 | 36.20 | 45.25 | **45.26** |

数据：`results/downstream/`

## 3. 医疗训练的层选择

验证集为医疗训练数据中分出的 1,152 条。先在通用域配置附近搜索（stage A），最佳落在边界时再隔层搜索（stage B）。

**Qwen3-4B**（共 8 组）

| 注入/分类层 | 19/22 | 19/24 | 21/22 | 21/24 | 17/22 | 17/24 | 17/20 | 19/20 |
|---|---|---|---|---|---|---|---|---|
| 验证 AUROC | 89.68 | 88.80 | 88.57 | 88.56 | 88.26 | 88.20 | 88.12 | 88.05 |

**Gemma2-2B**（共 43 组）

| 注入/分类层 | 12/18 | 12/16 | 12/20 | 14/16 | 11/25 |
|---|---|---|---|---|---|
| 验证 AUROC | 87.21 | 86.86 | 86.51 | 86.35 | 86.00 |

最终使用 Qwen3-4B 19/22、Gemma2-2B 12/18。数据：`results/layer_search/`（全部配置与 checkpoint）

## 4. 评测检查

- 规则无法提取选项的输出交给 GPT-4.1-mini 提取（只给选项，并要求引用原文），各方法 Micro 准确率变化最大 0.03。
- 随机抽取 1,000 条已判分输出，由 GPT-4.1-mini 判断对错，与规则判分一致率 99.50%，不一致样本均为 LLM 误判。

数据：`results/checks/`

## 结果文件

`results/index.csv` 列出每个数字和对应的数据文件。核对：

```bash
python verify.py
```

数据文件为 gzip 压缩，`gzip -dk <文件>` 解压查看。`results/training/` 为医疗训练数据与训练标注，`results/zero_shot_checkpoints/` 为零样本使用的通用域向量（Qwen3-4B 19/22、Gemma2-2B 14/18），`results/retrieval/` 为 MIRAGE 检索结果的段落编号。

## 代码

```
code/medical/     语料编码、检索、标注、训练数据、层搜索、打分、评测
code/rag_probe/   方法与基线（steering vector 训练、打分、生成）
run.sh
verify.py
```

检索与编码：Python 3.11，torch 2.5.1，transformers 4.44.2。训练、打分与生成：Python 3.12，torch 2.10.0，transformers 4.56.2，vLLM 0.19.0。
工作目录由 `MT_WORK` 指定（默认当前目录），需要 `data/mirage/benchmark.json`、MedRAG 的 PubMed 与教科书语料（`corpus/<name>/chunk/*.jsonl`）和 StatPearls 原始包（`STATPEARLS_TAR`）。标注与检查所需的 `KAKOU_API_KEY` / `KAKOU_BASE_URL`、`XTY_API_KEY` / `XTY_BASE_URL` 写在 `$MT_WORK/.env`。零样本 Ours 使用 `results/zero_shot_checkpoints/` 中的通用域 checkpoint（`CSV_RETRIEVED_DIR` 指向该目录），Utility CE 需要其模型权重（`CE_MODEL`），DUS 需要 `DUS_MODEL`。

```bash
bash run.sh corpus
bash run.sh retrieve
bash run.sh train_questions
bash run.sh label
bash run.sh train_data
bash run.sh score
for m in qwen3_4b gemma2b; do
  MODEL=$m bash run.sh score_ours
  MODEL=$m bash run.sh train
  MODEL=$m bash run.sh score_ours_med
  MODEL=$m bash run.sh generate
  MODEL=$m bash run.sh evaluate
done
bash run.sh checks
```
