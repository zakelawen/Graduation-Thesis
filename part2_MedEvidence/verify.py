import csv
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(HERE, "code", "medical"))
from eval_mcq import predict_letter


def load_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def auroc(y, s):
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(y)
    neg = len(y) - pos
    return (sum(r for r, t in zip(ranks, y) if t) - pos * (pos + 1) / 2) / (pos * neg)


def auprc(y, s):
    order = sorted(range(len(s)), key=lambda i: -s[i])
    pos = sum(y)
    tp = fp = 0
    ap = prev_recall = 0.0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and s[order[j]] == s[order[i]]:
            tp += y[order[j]]
            fp += 1 - y[order[j]]
            j += 1
        recall = tp / pos
        ap += (recall - prev_recall) * tp / (tp + fp)
        prev_recall = recall
        i = j
    return ap


def macro_f1(y, p):
    f1s = []
    for c in (0, 1):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f1s) / 2


def classification(values):
    labels = {r["id"]: r for r in load_gz(os.path.join(R, "classification/labels.jsonl.gz")) if r["label_name"] in ("relevant", "distracting")}
    scores = {r["id"]: r for r in load_gz(os.path.join(R, "classification/top1_scores.jsonl.gz"))}
    thresholds = json.load(open(os.path.join(R, "classification/thresholds.json"), encoding="utf-8"))
    ids = sorted(labels)
    for key, t in thresholds.items():
        sign = 1 if t["direction"] == "greater" else -1
        for ds in ["ALL"] + sorted({labels[i]["dataset"] for i in ids}):
            sub = [i for i in ids if ds == "ALL" or labels[i]["dataset"] == ds]
            y = [int(labels[i]["label_name"] == "relevant") for i in sub]
            s = [sign * scores[i]["scores"][key] for i in sub]
            if len(set(y)) < 2:
                continue
            values[("classification", f"{key}/{ds}", "auroc")] = 100 * auroc(y, s)
            if ds == "ALL":
                values[("classification", f"{key}/{ds}", "auprc")] = 100 * auprc(y, s)
                if t["threshold"] is not None:
                    p = [int(v >= sign * t["threshold"]) for v in s]
                    values[("classification", f"{key}/{ds}", "acc")] = 100 * sum(a == b for a, b in zip(y, p)) / len(y)
                    values[("classification", f"{key}/{ds}", "macro_f1")] = 100 * macro_f1(y, p)


def summarize(rows, correct):
    by = defaultdict(lambda: [0, 0])
    for r, c in zip(rows, correct):
        by[r["dataset"]][0] += 1
        by[r["dataset"]][1] += c
    out = {ds: 100 * c / n for ds, (n, c) in by.items()}
    out["ALL"] = 100 * sum(c for _, c in by.values()) / sum(n for n, _ in by.values())
    out["MACRO"] = sum(100 * c / n for n, c in by.values()) / len(by)
    return out


def downstream(values):
    cache = {}
    for r in load_gz(os.path.join(R, "checks/llm_answer_extraction.jsonl.gz")):
        if r["choice"] != "ERROR":
            cache[r["key"]] = r["choice"]
    d = os.path.join(R, "downstream")
    for name in sorted(os.listdir(d)):
        model, selector = name[: -len(".jsonl.gz")].split("__")
        rows = load_gz(os.path.join(d, name))
        letters = [predict_letter(r["prediction"], r["options"]) for r in rows]
        rule = summarize(rows, [l == r["answer"] for l, r in zip(letters, rows)])
        for ds, v in rule.items():
            values[("downstream", f"{model}/{selector}/{ds}", "acc")] = v
        values[("downstream", f"{model}/{selector}/ALL", "avg_docs")] = sum(len(r["selected_ranks"]) for r in rows) / len(rows)
        merged = []
        for l, r in zip(letters, rows):
            if l is None:
                l = cache.get(hashlib.sha1(f"v2|gpt-4.1-mini|{r['idx']}|{r['prediction']}".encode()).hexdigest())
            merged.append(l == r["answer"])
        llm = summarize(rows, merged)
        for ds in ("ALL", "MACRO"):
            values[("check_llm_extraction", f"{model}/{selector}/{ds}", "acc_rule_plus_llm")] = llm[ds]


def layer_search(values):
    for model in ("qwen3_4b", "gemma2b"):
        sweep = json.load(open(os.path.join(R, f"layer_search/{model}_sweep.json"), encoding="utf-8"))
        for key, e in sweep.items():
            values[("layer_search", f"{model}/{key}", "val_auroc")] = 100 * e["best_auroc"]


def judge(values):
    rows = load_gz(os.path.join(R, "checks/llm_judge_audit.jsonl.gz"))
    values[("check_llm_judge", "audit_1000", "agreement")] = 100 * sum(r["rule_correct"] == r["llm_correct"] for r in rows) / len(rows)


def main():
    values = {}
    for f in (classification, downstream, layer_search, judge):
        f(values)
    bad = 0
    rows = list(csv.DictReader(open(os.path.join(R, "index.csv"), encoding="utf-8")))
    for r in rows:
        key = (r["table"], r["row"], r["column"])
        got = values.get(key)
        if got is None or abs(got - float(r["value"])) > 0.006:
            bad += 1
            print(f"MISMATCH {key}: index {r['value']}, recomputed {got}")
    print(f"checked {len(rows)} numbers, {bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
