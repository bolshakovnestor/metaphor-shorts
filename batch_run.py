"""
Пакетна обробка корпусу Shorts для пілотного дослідження.

  python batch_run.py urls.txt              # аналіз + файл для ручної перевірки
  python batch_run.py urls.txt --preannotate  # + попередня розмітка іншою моделлю (gpt-5-mini)

Результат: results/*.json (кожне відео) та annotation.csv, де колонку gold_words
дослідник перевіряє вручну (спершу вона заповнена попередньою розміткою).
"""
import csv
import json
import os
import sys
import time

import pipeline as P

urls = [u.strip() for u in open(sys.argv[1], encoding="utf-8") if u.strip() and not u.startswith("#")]
pre = "--preannotate" in sys.argv
client = P.get_client()
os.makedirs("results", exist_ok=True)
rows = []
for n, url in enumerate(urls, 1):
    print(f"[{n}/{len(urls)}] {url}")
    try:
        rep = P.analyze_url(client, url, log=print)
    except Exception as e:
        print("   ПРОПУЩЕНО:", e)
        continue
    vid = rep["source"].get("id") or f"video{n}"
    with open(f"results/{vid}.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    for s in rep["sentences"]:
        pa = P.preannotate(client, s["text"]) if pre else ""
        rows.append({"video_id": vid, "url": url, "sentence_id": s["id"], "start": P.fmt_time(s["start"]),
                     "sentence": s["text"], "model_words": ", ".join(s["verified"]) or "none",
                     "model_phrases": " | ".join(f'{p["phrase"]} [{"C" if p.get("conventionality") == "conventional" else "N"}]' for p in s["phrases"]),
                     "preannotation": pa, "gold_words": pa, "checked": "", "comment": ""})
    time.sleep(0.5)

with open("annotation.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["video_id"])
    w.writeheader(); w.writerows(rows)
print(f"Готово: {len(rows)} речень -> annotation.csv")
