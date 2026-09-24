"""
Конвеєр ідентифікації метафор у субтитрах YouTube Shorts.

Етапи:
  1. Аудіо з відео (yt-dlp) -> транскрипт із тайм-кодами (Whisper, OpenAI API)
  2. Сегментація на речення
  3. Детектор: доналаштована gpt-4o (VUA, 1 500 речень) -> слова-кандидати
  4. Фільтр: лише повнозначні слова, що справді є в реченні
  5. Верифікація за спрощеною MIP + об'єднання у фрази (gpt-4o, JSON-схема)
  6. Контроль: фраза має бути дослівним фрагментом речення
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# ------------------------------------------------------------------ config
DETECTOR_MODEL = os.getenv("DETECTOR_MODEL", "ft:gpt-4o-2024-08-06:personal::BvA0wGRA")
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "gpt-4o-2024-08-06")   # фіксований знімок
ASR_MODEL = os.getenv("ASR_MODEL", "whisper-1")
PREANNOTATOR_MODEL = os.getenv("PREANNOTATOR_MODEL", "gpt-5-mini")  # інша модель для сріблого стандарту
# Self-consistency (Wang et al., 2022): скільки разів запускати верифікацію.
# 1 — швидко (демонстрація), 2 — стабільніше (корпус): лишаємо тільки те, що підтвердилося в усіх проходах.
VERIFIER_PASSES = int(os.getenv("VERIFIER_PASSES", "1"))

# Орієнтовні тарифи, USD за 1M токенів (вхід, вихід). Перевір актуальні на сайті OpenAI.
PRICES = {DETECTOR_MODEL: (3.75, 15.0), VERIFIER_MODEL: (2.50, 10.0)}
ASR_PRICE_PER_MIN = 0.006

# Той самий system prompt, що й у навчальному JSONL: без нього модель відповідає прозою.
DETECTOR_SYSTEM = (
    "You are a linguistic analyzer. Your task is to identify metaphorical words in a sentence. "
    "List them separated by a comma, or output 'none' if there are no metaphors."
)

# Службові слова (закритий клас). Узгоджено зі скриптом оцінювання на VUA.
FUNCTION_WORDS = set("""
a an the this that these those some any each every no all both either neither
i me my mine you your yours he him his she her hers it its we us our ours they them their theirs
one oneself itself himself herself themselves ourselves yourself myself
who whom whose which what whatever whoever
and or but nor so yet if then than because although though while whereas unless until till since as whether
of in on at to from by with without within into onto upon over under above below between among through
throughout across along around about against toward towards beyond behind beneath beside besides near
off out up down after before during per via like unlike despite inside outside amid
be am is are was were been being have has had having do does did doing done
will would shall should can could may might must ought
not n't there here very just also only even too quite rather such own same more most less least much many few
several other another
i'm you're he's she's it's we're they're i've you've we've they've i'll you'll it'll i'd you'd
that's there's what's let's don't doesn't didn't can't won't isn't aren't wasn't weren't gonna wanna gotta
""".split())

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def get_client(api_key: str | None = None) -> OpenAI:
    return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))


# ------------------------------------------------------------------ 1. input
def download_audio(url: str, workdir: str) -> tuple[str, dict]:
    """Завантажує лише аудіодоріжку (без ffmpeg). Повертає шлях і метадані."""
    import yt_dlp

    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": os.path.join(workdir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "max_filesize": 24 * 1024 * 1024,  # ліміт API транскрибації — 25 МБ
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = info.get("requested_downloads", [{}])[0].get("filepath") or ydl.prepare_filename(info)
    meta = {k: info.get(k) for k in ("id", "title", "uploader", "duration", "webpage_url", "upload_date")}
    return path, meta


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def transcribe(client: OpenAI, audio_path: str) -> tuple[str, list[dict], float]:
    """Whisper -> (текст, сегменти з тайм-кодами, тривалість у секундах)."""
    with open(audio_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model=ASR_MODEL, file=f, response_format="verbose_json", language="en"
        )
    segs = [
        {"start": float(_get(s, "start", 0)), "end": float(_get(s, "end", 0)), "text": str(_get(s, "text", "")).strip()}
        for s in (_get(tr, "segments") or [])
    ]
    text = str(_get(tr, "text", "")).strip()
    if not segs and text:
        segs = [{"start": 0.0, "end": 0.0, "text": text}]
    duration = float(_get(tr, "duration", 0) or (segs[-1]["end"] if segs else 0))
    return text, segs, duration


def parse_srt(raw: str) -> list[dict]:
    """.srt / .vtt -> сегменти. Прибирає номери, тайм-коди, теги й дублі рядків."""
    t_re = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->")
    segs, cur_start, buf = [], None, []

    def flush():
        if cur_start is not None and buf:
            segs.append({"start": cur_start, "end": cur_start, "text": " ".join(buf)})

    for line in raw.splitlines():
        line = line.strip().lstrip("\ufeff")
        m = t_re.search(line)
        if m:
            flush()
            h, mi, s, ms = map(int, m.groups())
            cur_start, buf = h * 3600 + mi * 60 + s + ms / 1000, []
        elif not line or line.isdigit() or line.upper().startswith("WEBVTT"):
            continue
        else:
            clean = re.sub(r"<[^>]+>|\{[^}]+\}", "", line).strip()
            if clean and (not segs or clean != segs[-1]["text"]) and clean not in buf:
                buf.append(clean)
    flush()
    return segs


# ------------------------------------------------------------------ 2. segmentation
SENT_SPLIT = re.compile(r"(?<=[.!?…])[\"”’)]*\s+")
# ініціали й типові скорочення, після яких речення не закінчується
ABBREV_END = re.compile(r"(?:\b[A-Z]\.|\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|No|Mt)\.)$")


def segments_to_sentences(segs: list[dict], max_words: int = 40) -> list[dict]:
    """Склеює сегменти, ділить на речення, кожному дає тайм-код початку."""
    full, offsets = "", []
    for s in segs:
        if not s["text"]:
            continue
        if full:
            full += " "
        offsets.append((len(full), s["start"]))
        full += s["text"]
    full = re.sub(r"\s+", " ", full).strip()

    def time_at(pos: int) -> float:
        t = 0.0
        for off, st in offsets:
            if off <= pos:
                t = st
        return t

    chunks = []
    for ch in SENT_SPLIT.split(full):
        ch = ch.strip()
        if chunks and ABBREV_END.search(chunks[-1]):
            chunks[-1] += " " + ch          # "Michael B." + "Jordan for me." -> одне речення
        elif ch:
            chunks.append(ch)
    out, pos = [], 0
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        start_pos = full.find(chunk, pos)
        pos = start_pos + len(chunk) if start_pos >= 0 else pos
        words = chunk.split()
        # субтитри без пунктуації: ріжемо на вікна, щоб не подавати моделі «простирадло»
        for i in range(0, len(words), max_words):
            piece = " ".join(words[i:i + max_words])
            out.append({"text": piece, "start": time_at(max(start_pos, 0))})
    return out


def text_to_sentences(text: str) -> list[dict]:
    return segments_to_sentences([{"start": 0.0, "end": 0.0, "text": text}])


# ------------------------------------------------------------------ 3–4. detector
def run_detector(client: OpenAI, sentence: str) -> tuple[str, dict]:
    r = client.chat.completions.create(
        model=DETECTOR_MODEL,
        messages=[{"role": "system", "content": DETECTOR_SYSTEM},
                  {"role": "user", "content": f"Sentence: {sentence}"}],
        temperature=0,
        max_tokens=120,
    )
    raw = (r.choices[0].message.content or "").strip()
    return raw.splitlines()[0] if raw else "none", _usage(DETECTOR_MODEL, r)


def parse_candidates(raw: str, sentence: str) -> list[str]:
    """Лише повнозначні слова, які реально є в реченні (захист від вигаданих слів)."""
    if not raw or raw.lower().startswith("none"):
        return []
    in_sentence = {w.lower().replace("’", "'") for w in WORD_RE.findall(sentence)}
    out = []
    for p in raw.split(","):
        w = re.sub(r"^[^A-Za-z'’-]+|[^A-Za-z'’-]+$", "", p.strip()).lower().replace("’", "'")
        if w and w not in FUNCTION_WORDS and w in in_sentence and w not in out:
            out.append(w)
    return out


# ------------------------------------------------------------------ 5. verifier
VERIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["words", "phrases"],
    "properties": {
        "words": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["word", "contextual_meaning", "basic_meaning", "is_metaphor"],
            "properties": {
                "word": {"type": "string"},
                "contextual_meaning": {"type": "string"},
                "basic_meaning": {"type": "string"},
                "is_metaphor": {"type": "boolean"}}}},
        "phrases": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["phrase", "triggers", "conventionality", "source_domain", "target_domain", "explanation_uk"],
            "properties": {
                "phrase": {"type": "string"},
                "conventionality": {"type": "string", "enum": ["conventional", "creative"]},
                "triggers": {"type": "array", "items": {"type": "string"}},
                "source_domain": {"type": "string"},
                "target_domain": {"type": "string"},
                "explanation_uk": {"type": "string"}}}},
    },
}

VERIFY_SYSTEM = """You verify metaphor candidates with a simplified MIP (Pragglejaz Group, 2007).
For EACH candidate word:
 1) state its contextual meaning in THIS sentence;
 2) state its more basic meaning (more concrete, bodily, or historically older);
 3) is_metaphor = true only if the two meanings contrast AND the contextual one is understood by comparison with the basic one.
    NOT metaphors: metonymy, literal use, proper names, plain intensifiers, discourse markers
    ("See,", "look,", "you know"), grammaticalised quantifiers/hedges ("a lot", "a bit", "kind of").
Then group the verified metaphorical words into metaphorical PHRASES:
 - every phrase must be an EXACT, character-for-character substring of SENTENCE (keep original spelling and case);
 - keep phrases short (2-8 words) and include at least one verified word in "triggers";
 - conventionality: "conventional" if the metaphorical sense is lexicalised and listed in a standard dictionary
   (e.g. "get it" = understand, "came from" = originated); "creative" if it is a novel or deliberately built image
   (e.g. "my motivation ghosted me", "attention turned into confetti");
 - source_domain / target_domain: short conceptual labels in English capitals (e.g. JOURNEY / LIFE);
 - explanation_uk: one short sentence in Ukrainian explaining the mapping.
Use CONTEXT only to understand meaning; never take phrases from CONTEXT.
If no candidate is metaphorical, return empty lists."""


def run_verifier(client: OpenAI, sentence: str, candidates: list[str], context: str) -> tuple[dict, dict]:
    if not candidates:
        return {"words": [], "phrases": []}, {}
    user = f"CONTEXT: {context or '-'}\nSENTENCE: {sentence}\nCANDIDATES: {', '.join(candidates)}"
    r = client.chat.completions.create(
        model=VERIFIER_MODEL,
        messages=[{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}],
        temperature=0,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "mip_verification", "strict": True, "schema": VERIFY_SCHEMA}},
    )
    try:
        data = json.loads(r.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        data = {"words": [], "phrases": []}
    return data, _usage(VERIFIER_MODEL, r)


def validate(sentence: str, candidates: list[str], data: dict) -> tuple[list[str], list[dict]]:
    """6. Відкидає все, що не є дослівним фрагментом речення або не спирається на кандидата."""
    cand = set(candidates)
    verified = [w["word"].lower() for w in data.get("words", [])
                if w.get("is_metaphor") and w.get("word", "").lower() in cand]
    low = sentence.lower()
    phrases = []
    for p in data.get("phrases", []):
        ph = p.get("phrase", "").strip().strip("\"'“”")
        idx = low.find(ph.lower()) if ph else -1
        trig = [t.lower() for t in p.get("triggers", []) if t.lower() in verified]
        if idx < 0 or not trig:
            continue
        p = dict(p, phrase=sentence[idx: idx + len(ph)], triggers=trig, span=[idx, idx + len(ph)])
        phrases.append(p)
    return verified, phrases


# ------------------------------------------------------------------ orchestration
def _borderline(p: dict) -> bool:
    a = p.get("agreement", "")
    return "/" in a and a.split("/")[0] != a.split("/")[1]


def _usage(model: str, resp) -> dict:
    u = getattr(resp, "usage", None)
    return {"model": model, "in": getattr(u, "prompt_tokens", 0) or 0, "out": getattr(u, "completion_tokens", 0) or 0}


def _merge_passes(runs: list[tuple[list[str], list[dict]]]) -> tuple[list[str], list[dict]]:
    """Self-consistency (Wang et al., 2022): рішення більшістю проходів.

    Слово / фразу залишаємо, якщо її підтвердила більшість проходів. Поле agreement
    показує, у скількох проходах із N вона з'явилася: N/N — стабільна знахідка,
    менше — «сіра зона» метафоричності, яку варто перевіряти вручну.
    """
    n = len(runs)
    if not n:
        return [], []
    need = n // 2 + 1
    wcount, pcount, pdata = {}, {}, {}
    for words, phrases in runs:
        for w in set(words):
            wcount[w] = wcount.get(w, 0) + 1
        for p in phrases:
            k = p["phrase"].lower()
            pcount[k] = pcount.get(k, 0) + 1
            pdata.setdefault(k, []).append(p)
    verified = [w for w, c in wcount.items() if c >= need]
    phrases = []
    for k, c in pcount.items():
        if c < need:
            continue
        variants = pdata[k]
        conv = [v.get("conventionality") for v in variants]
        p = dict(variants[0], agreement=f"{c}/{n}",
                 conventionality=max(set(conv), key=conv.count) if conv else "conventional")
        if all(t in verified for t in p["triggers"]):
            phrases.append(p)
    phrases.sort(key=lambda p: p["span"][0])
    return verified, phrases


def analyze_sentence(client: OpenAI, sentences: list[dict], i: int) -> dict:
    s = sentences[i]["text"]
    context = " ".join(x["text"] for x in sentences[max(0, i - 1): i + 2] if x["text"] != s)
    usage = []
    try:
        raw, u1 = run_detector(client, s); usage.append(u1)
        cands = parse_candidates(raw, s)
        runs, data = [], {"words": []}
        for _ in range(max(VERIFIER_PASSES, 1)):
            d, u2 = run_verifier(client, s, cands, context)
            if u2:
                usage.append(u2)
            data = data if data.get("words") else d
            runs.append(validate(s, cands, d))
        verified, phrases = _merge_passes(runs)
        err = None
    except Exception as e:  # один збій не валить увесь аналіз
        raw, cands, data, verified, phrases = "", [], {"words": []}, [], []
        err = f"{type(e).__name__}: {e}"
    return {"id": i + 1, "start": sentences[i]["start"], "text": s, "detector_raw": raw,
            "candidates": cands, "verification": data.get("words", []), "verified": verified,
            "phrases": phrases, "passes": VERIFIER_PASSES, "error": err, "usage": usage}


def analyze_sentences(client: OpenAI, sentences: list[dict], workers: int = 4, progress=None) -> list[dict]:
    results = [None] * len(sentences)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(analyze_sentence, client, sentences, i): i for i in range(len(sentences))}
        for n, f in enumerate(as_completed(futs), 1):
            results[futs[f]] = f.result()
            if progress:
                progress(n / len(sentences))
    return results


def build_report(source: dict, sentences_out: list[dict], transcript: str, audio_minutes: float = 0.0) -> dict:
    n_words = sum(len(WORD_RE.findall(s["text"])) for s in sentences_out)
    n_phr = sum(len(s["phrases"]) for s in sentences_out)
    n_cre = sum(1 for s in sentences_out for p in s["phrases"] if p.get("conventionality") == "creative")
    n_border = sum(1 for s in sentences_out for p in s["phrases"] if _borderline(p))
    cost = audio_minutes * ASR_PRICE_PER_MIN
    for s in sentences_out:
        for u in s["usage"]:
            pin, pout = PRICES.get(u["model"], (0, 0))
            cost += u["in"] / 1e6 * pin + u["out"] / 1e6 * pout
    return {
        "source": source,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "models": {"asr": ASR_MODEL, "detector": DETECTOR_MODEL, "verifier": VERIFIER_MODEL},
        "transcript": transcript,
        "sentences": sentences_out,
        "stats": {"sentences": len(sentences_out), "words": n_words, "phrases": n_phr,
                  "verified_words": sum(len(s["verified"]) for s in sentences_out),
                  "creative": n_cre, "conventional": n_phr - n_cre,
                  "density_per_100_words": round(100 * n_phr / n_words, 2) if n_words else 0.0,
                  "creative_per_100_words": round(100 * n_cre / n_words, 2) if n_words else 0.0,
                  "errors": sum(1 for s in sentences_out if s["error"]),
                  "borderline": n_border, "stable": n_phr - n_border,
                  "verifier_passes": VERIFIER_PASSES,
                  "cost_usd_estimate": round(cost, 4)},
    }


def analyze_url(client: OpenAI, url: str, progress=None, log=None) -> dict:
    log = log or (lambda *_: None)
    with tempfile.TemporaryDirectory() as tmp:
        log("Завантаження аудіо…")
        path, meta = download_audio(url, tmp)
        log("Транскрибація (Whisper)…")
        text, segs, duration = transcribe(client, path)
    sents = segments_to_sentences(segs)
    log(f"Аналіз {len(sents)} речень…")
    out = analyze_sentences(client, sents, progress=progress)
    return build_report({"type": "youtube", "url": url, **meta}, out, text, duration / 60)


def analyze_text(client: OpenAI, text: str, progress=None) -> dict:
    sents = text_to_sentences(text)
    return build_report({"type": "text"}, analyze_sentences(client, sents, progress=progress), text)


def analyze_srt(client: OpenAI, raw: str, name: str = "", progress=None) -> dict:
    segs = parse_srt(raw)
    sents = segments_to_sentences(segs)
    transcript = " ".join(s["text"] for s in segs)
    return build_report({"type": "srt", "file": name}, analyze_sentences(client, sents, progress=progress), transcript)


# ------------------------------------------------------------------ silver standard helper
PREANNOTATE_PROMPT = """Apply MIP (Pragglejaz Group, 2007) to the sentence below, considering ONLY content words
(nouns, verbs, adjectives, adverbs). A word is metaphorical if its contextual meaning contrasts with a more basic
meaning and is understood by comparison with it. Output ONLY the metaphorical words, comma-separated, or: none

Sentence: {sentence}"""


def preannotate(client: OpenAI, sentence: str) -> str:
    """Попередня розмітка ІНШОЮ моделлю — щоб еталон не збігався з оцінюваним детектором."""
    r = client.chat.completions.create(model=PREANNOTATOR_MODEL,
                                       messages=[{"role": "user", "content": PREANNOTATE_PROMPT.format(sentence=sentence)}])
    raw = (r.choices[0].message.content or "none").strip().splitlines()[0]
    return ", ".join(parse_candidates(raw, sentence)) or "none"


# ------------------------------------------------------------------ presentation helpers
def fmt_time(sec: float) -> str:
    sec = int(sec or 0)
    return f"{sec // 60}:{sec % 60:02d}"


def highlight_html(sentence: str, phrases: list[dict]) -> str:
    import html
    spans = sorted((p["span"][0], p["span"][1], i) for i, p in enumerate(phrases) if "span" in p)
    merged = []
    for a, b, i in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b, i])
    out, pos = [], 0
    for a, b, i in merged:
        out.append(html.escape(sentence[pos:a]))
        p = phrases[i]
        conv = p.get("conventionality") == "conventional"
        cls = "mb" if _borderline(p) else ("mc" if conv else "mt")
        tip = html.escape(f'{"конвенційна" if conv else "креативна"}'
                          f'{", спірна " + p["agreement"] if _borderline(p) else ""}: '
                          f'{p["source_domain"]} → {p["target_domain"]}')
        out.append(f'<mark class="{cls}" title="{tip}">{html.escape(sentence[a:b])}</mark>')
        pos = b
    out.append(html.escape(sentence[pos:]))
    return "".join(out)


def report_rows(report: dict) -> list[dict]:
    rows = []
    for s in report["sentences"]:
        for p in s["phrases"]:
            rows.append({"Час": fmt_time(s["start"]), "Фраза": p["phrase"],
                         "Тип": "конвенційна" if p.get("conventionality") == "conventional" else "креативна",
                         "Пояснення": p["explanation_uk"],
                         "Відображення": f'{p["source_domain"]} → {p["target_domain"]}',
                         "Згода": p.get("agreement", "1/1"),
                         "Тригери": ", ".join(p["triggers"]), "Речення №": s["id"], "Речення": s["text"]})
    return rows
