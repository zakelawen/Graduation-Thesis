import csv
import gzip
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))


def count(path, field):
    n = c = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
                c += str(json.loads(line).get(field)) in ("1", "1.0", "True")
    return c, n


def notebook_text(path, k):
    cell = json.load(open(path, encoding="utf-8"))["cells"][k - 1]
    text = ""
    for o in cell.get("outputs", []):
        t = o.get("text") or "".join(o.get("data", {}).get("text/plain", []))
        text += "".join(t) if isinstance(t, list) else t
    return text


def check(row):
    value = float(row["value"])
    m = re.match(r"(.+\.jsonl\.gz) \((\w+)\)$", row["result"])
    if m:
        c, n = count(os.path.join(ROOT, m.group(1)), m.group(2))
        return abs(100 * c / n - value) < 0.005, f"{c}/{n}"
    m = re.match(r"(.+\.ipynb) cell (\d+)$", row["result"])
    if m:
        ok = f"{value:.2f}%" in notebook_text(os.path.join(ROOT, m.group(1)), int(m.group(2)))
        if row["data_file"]:
            ok = ok and os.path.exists(os.path.join(ROOT, row["data_file"]))
        return ok, f"{row['correct']}/{row['total']}"
    return None, row["result"]


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "index.csv"), encoding="utf-8-sig")))
    passed = failed = 0
    for row in rows:
        ok, detail = check(row)
        status = "OK" if ok else ("--" if ok is None else "FAIL")
        passed += ok is True
        failed += ok is False
        print(f"{status:4s} {row['table']:8s} {row['method']:16s} {row['dataset']:8s} {row['model']:12s} {row['value']:>6s}  {detail}")
    print(f"\n{passed} OK, {failed} FAIL, {len(rows) - passed - failed} without data file")


if __name__ == "__main__":
    main()
