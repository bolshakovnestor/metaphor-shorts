# Метафори в субтитрах YouTube Shorts — прототип

Посилання на Shorts → аудіо (yt-dlp) → транскрипт (Whisper) → речення →
детектор (доналаштована gpt-4o) → фільтр службових слів → верифікація MIP і фрази (gpt-4o) → звіт.

## Запуск на своєму ноутбуці (для демонстрації)

1. Встанови Python 3.10+ (python.org, під час встановлення познач «Add Python to PATH»).
2. У папці проєкту відкрий термінал:
   ```
   python -m venv .venv
   .venv\Scripts\activate        (Windows)   |   source .venv/bin/activate   (macOS/Linux)
   pip install -r requirements.txt
   ```
3. Скопіюй `.streamlit/secrets.toml.example` у `.streamlit/secrets.toml` і встав **новий** ключ OpenAI.
4. `streamlit run app.py` — відкриється браузер на http://localhost:8501

Якщо YouTube перестав віддавати відео: `pip install -U yt-dlp` (YouTube часто змінює захист;
новим версіям yt-dlp може знадобитися JavaScript-рушій Deno — дивись README yt-dlp).

## Офлайн-режим
Кожен аналіз зберігається в `results/`. Вкладка «Збережені аналізи» відкриває їх без інтернету.
Перед виступом прожени 3–5 Shorts і закоміть `results/` — демо не залежатиме від мережі.

## Хмара (запасний варіант)
Завантаж репозиторій на GitHub → share.streamlit.io → New app → `app.py`
→ Advanced settings → Secrets: `OPENAI_API_KEY = "sk-..."`.
На хмарних IP YouTube часто блокує завантаження — там використовуй вкладки «Текст» і «Файл .srt».

## Пілотний корпус
```
python batch_run.py urls.txt --preannotate     # аналіз + попередня розмітка gpt-5-mini
```
Відкрий `annotation.csv` в Excel: у кожному рядку виправ `gold_words` (лише повнозначні слова через кому
або `none`) і постав `1` у `checked`. Потім:
```
python eval_silver.py annotation.csv           # Precision / Recall / F1 + % твоїх виправлень
```

## Параметри (змінні середовища)
`DETECTOR_MODEL`, `VERIFIER_MODEL` (за замовчуванням фіксований знімок gpt-4o-2024-08-06), `ASR_MODEL`, `PREANNOTATOR_MODEL`.
Ключ ніколи не пиши в код і не комітить `secrets.toml` (він у `.gitignore`).
