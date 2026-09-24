"""
Оцінка на перевіреному еталоні Shorts:  python eval_silver.py annotation.csv
Рахує лише рядки з checked = 1 (перевірено дослідником). Одиниця — повнозначне слово.
Також показує, скільки міток попередньої розмітки дослідник змінив.
"""
import sys

import pandas as pd

import pipeline as P


def words(x, sentence):
    x = "" if pd.isna(x) else str(x)
    return set(P.parse_candidates(x, sentence))


df = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
df = df[df["checked"].astype(str).str.strip() == "1"]
tp = fp = fn = changed = pre_total = 0
for _, r in df.iterrows():
    g, m, pa = words(r["gold_words"], r["sentence"]), words(r["model_words"], r["sentence"]), words(r["preannotation"], r["sentence"])
    tp += len(g & m); fp += len(m - g); fn += len(g - m)
    changed += len(pa ^ g); pre_total += len(pa | g)
p = tp / (tp + fp) if tp + fp else 0
r_ = tp / (tp + fn) if tp + fn else 0
f1 = 2 * p * r_ / (p + r_) if p + r_ else 0
print(f"Перевірених речень: {len(df)}")
print(f"TP={tp} FP={fp} FN={fn}  Precision={p:.3f}  Recall={r_:.3f}  F1={f1:.3f}")
if pre_total:
    print(f"Змінено дослідником: {changed} з {pre_total} міток попередньої розмітки ({100*changed/pre_total:.1f} %)")
