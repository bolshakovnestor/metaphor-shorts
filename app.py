"""Streamlit-інтерфейс: посилання на Shorts / текст / .srt -> метафоричні фрази."""
import glob
import json
import os
import re

import pandas as pd
import streamlit as st

import pipeline as P

st.set_page_config(page_title="Метафори в Shorts", page_icon="🖍️", layout="wide")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


DEFAULT_PASSWORD = "24092026"   # можна перевизначити ключем APP_PASSWORD у secrets


def check_password() -> bool:
    """Простий вхід за паролем, щоб публічне посилання не витрачало квоту API."""
    try:
        pwd = st.secrets.get("APP_PASSWORD", DEFAULT_PASSWORD)
    except Exception:
        pwd = DEFAULT_PASSWORD
    if not pwd or st.session_state.get("auth"):
        return True
    st.markdown("### Метафори в субтитрах YouTube Shorts")
    st.caption("Доступ за паролем. Пароль надає автор дослідження.")
    with st.form("login"):
        entered = st.text_input("Пароль", type="password")
        if st.form_submit_button("Увійти"):
            if entered == pwd:
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("Неправильний пароль")
    return False


if not check_password():
    st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Literata:opsz,wght@7..72,400;7..72,600&display=swap');
:root { --ink:#17212B; --muted:#5B6B7B; --line:#DDE2E8; --marker:#FFE27A; --marker-edge:#D9A400; }
html, body, [class*="css"], .stMarkdown, .stTextInput, .stButton { font-family:'IBM Plex Sans', sans-serif; color:var(--ink); }
h1 { font-family:'Literata', serif; font-weight:600; letter-spacing:-0.01em; }
.lead { color:var(--muted); font-size:1.02rem; max-width:62ch; margin-top:-0.6rem; }
.tr-row { display:grid; grid-template-columns:3.2rem 1fr; gap:.8rem; padding:.55rem 0; border-bottom:1px solid var(--line); }
.tr-time { color:var(--muted); font-variant-numeric:tabular-nums; font-size:.85rem; padding-top:.2rem; }
.tr-text { font-family:'Literata', serif; font-size:1.12rem; line-height:1.65; max-width:75ch; }
mark.mt { background:linear-gradient(transparent 38%, var(--marker) 38%); border-bottom:2px solid var(--marker-edge);
          padding:0 .08em; color:inherit; cursor:help; }
mark.mb { background:none; border-bottom:2px dashed #C77B3A; padding:0 .08em; color:inherit; cursor:help; }
mark.mc { background:none; border-bottom:2px dotted #8A99A8; padding:0 .08em; color:inherit; cursor:help; }
.legend { color:var(--muted); font-size:.88rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def get_key() -> str | None:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY") or st.session_state.get("manual_key")


def video_embed_url(url: str) -> str:
    m = re.search(r"(?:shorts/|v=|youtu\.be/)([\w-]{11})", url or "")
    return f"https://www.youtube.com/watch?v={m.group(1)}" if m else url


def save_report(rep: dict) -> str:
    src = rep["source"]
    slug = src.get("id") or re.sub(r"\W+", "_", src.get("file", "") or "text")[:30]
    path = os.path.join(RESULTS_DIR, f"{rep['created'].replace(':', '-')}_{slug}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    return path


def run(fn, *args):
    key = get_key()
    if not key:
        st.error("Немає ключа OpenAI. Додай його в .streamlit/secrets.toml або в поле на бічній панелі.")
        return
    client = P.get_client(key)
    bar = st.progress(0.0, text="Підготовка…")
    status = st.empty()
    try:
        if fn is P.analyze_url:
            rep = fn(client, *args, progress=lambda x: bar.progress(x, text="Аналіз речень…"),
                     log=lambda m: status.info(m))
        else:
            rep = fn(client, *args, progress=lambda x: bar.progress(x, text="Аналіз речень…"))
    except Exception as e:
        bar.empty(); status.empty()
        msg = str(e)
        blocked = any(s in msg for s in ("403", "Forbidden", "Sign in", "not a bot", "bot")) \
            or "DownloadError" in type(e).__name__
        if blocked:
            st.error("**Завантажити це відео з сервера не вдалося.** "
                     "YouTube обмежує доступ для IP-адрес хмарних дата-центрів, "
                     "на яких працює ця онлайн-версія.")
            st.info("Що можна зробити просто зараз:\n\n"
                    "1. Відкрити вкладку **«Збережені аналізи»** — там наведено готові розбори "
                    "реальних відео YouTube Shorts.\n"
                    "2. Скористатися вкладкою **«Текст»** і вставити англомовний фрагмент.\n"
                    "3. Завантажити файл субтитрів у вкладці **«Файл .srt / .vtt»**.\n\n"
                    "Повний цикл із завантаженням відео за посиланням працює в локальній версії "
                    "застосунку і демонструється окремо.")
        else:
            st.error(f"Помилка: {type(e).__name__}: {msg}")
        return
    bar.empty(); status.empty()
    save_report(rep)
    st.session_state["report"] = rep


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.subheader("Налаштування")
    if not get_key():
        st.session_state["manual_key"] = st.text_input("OpenAI API key", type="password")
    show_steps = st.toggle("Показувати проміжні етапи", value=True)
    show_conv = st.toggle("Показувати конвенційні метафори", value=True,
                          help="Лексикалізовані значення на кшталт get it = understand, came from = originated")
    st.markdown(
        f"<div class='legend'>Детектор: <code>{P.DETECTOR_MODEL}</code><br>"
        f"Верифікатор: <code>{P.VERIFIER_MODEL}</code><br>Транскрибація: <code>{P.ASR_MODEL}</code></div>",
        unsafe_allow_html=True)

# ------------------------------------------------------------------ header + input
st.title("Метафори в субтитрах YouTube Shorts")
st.markdown("<p class='lead'>Доналаштована на VU Amsterdam Metaphor Corpus модель gpt-4o пропонує слова-кандидати, "
            "окремий етап перевіряє їх за процедурою MIP і збирає в метафоричні фрази.</p>", unsafe_allow_html=True)

t_url, t_text, t_srt, t_saved = st.tabs(["Посилання на Shorts", "Текст", "Файл .srt / .vtt", "Збережені аналізи"])

with t_url:
    url = st.text_input("Посилання", placeholder="https://www.youtube.com/shorts/…")
    st.caption("У цій онлайн-версії завантаження відео з YouTube може бути недоступним через "
               "обмеження для серверних IP-адрес. Готові розбори реальних відео — у вкладці "
               "«Збережені аналізи».")
    if st.button("Проаналізувати відео", type="primary", disabled=not url):
        run(P.analyze_url, url.strip())

with t_text:
    txt = st.text_area("Англійський текст", height=140,
                       placeholder="Anxiety was driving the car, so I grabbed the wheel and hit the brakes.")
    if st.button("Проаналізувати текст", type="primary", disabled=not txt.strip()):
        run(P.analyze_text, txt.strip())

with t_srt:
    up = st.file_uploader("Файл субтитрів", type=["srt", "vtt"])
    if st.button("Проаналізувати субтитри", type="primary", disabled=up is None):
        run(P.analyze_srt, up.getvalue().decode("utf-8", errors="ignore"), up.name)

with t_saved:
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")), reverse=True)
    if not files:
        st.caption("Ще немає збережених аналізів. Кожен запуск зберігається автоматично — це офлайн-режим для демонстрації.")
    else:
        def label(p):
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                src = d["source"]
                name = src.get("title") or src.get("file") or d["sentences"][0]["text"][:50]
                return f"{d['created'][:16].replace('T', ' ')}  |  {name}"
            except Exception:
                return os.path.basename(p)
        pick = st.selectbox("Аналіз", files, format_func=label)
        if st.button("Відкрити"):
            with open(pick, encoding="utf-8") as f:
                st.session_state["report"] = json.load(f)

# ------------------------------------------------------------------ report
rep = st.session_state.get("report")
if rep:
    st.divider()
    src, stt = rep["source"], rep["stats"]
    if src.get("type") == "youtube":
        st.subheader(src.get("title") or "Відео")
        st.caption(f'{src.get("uploader") or ""}  |  {src.get("duration") or "?"} с  |  {src.get("webpage_url") or src.get("url")}')
    m = st.columns(5)
    m[0].metric("Речень", stt["sentences"])
    m[1].metric("Метафоричних фраз", stt["phrases"])
    m[2].metric("з них креативних", stt.get("creative", "—"))
    m[3].metric("Фраз на 100 слів", stt["density_per_100_words"])
    m[4].metric("Вартість, $", f'{stt["cost_usd_estimate"]:.3f}')
    if stt.get("borderline"):
        st.caption(f'Стабільних знахідок: {stt.get("stable")}, спірних (підтвердила лише частина проходів): '
                   f'{stt["borderline"]}. Проходів верифікації: {stt.get("verifier_passes", 1)}.')
    if stt.get("errors"):
        st.warning(f'Речень з помилкою API: {stt["errors"]}. Їх пропущено.')

    def visible(ps):
        return [p for p in ps if show_conv or p.get("conventionality") != "conventional"]

    if src.get("type") == "youtube":
        c1, box = st.columns([1, 2], gap="large")
        with c1:
            st.video(video_embed_url(src.get("webpage_url") or src.get("url")))
    else:
        box = st.container()
    with box:
        st.markdown("<div class='legend'>Жовтий маркер — креативна метафора, пунктир — конвенційна. "
                    "Наведи курсор, щоб побачити відображення «джерело → мета».</div>", unsafe_allow_html=True)
        rows_html = "".join(
            f"<div class='tr-row'><div class='tr-time'>{P.fmt_time(s['start'])}</div>"
            f"<div class='tr-text'>{P.highlight_html(s['text'], visible(s['phrases']))}</div></div>"
            for s in rep["sentences"])
        st.markdown(f"<div style='max-height:620px;overflow-y:auto;padding-right:.5rem'>{rows_html}</div>",
                    unsafe_allow_html=True)

    rows = P.report_rows(rep)
    st.subheader("Виявлені метафори")
    if not show_conv:
        rows = [r for r in rows if r["Тип"] != "конвенційна"]
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df.drop(columns=["Речення"]), width="stretch", hide_index=True)
    else:
        df = pd.DataFrame()
        st.info("Метафор не виявлено.")

    if show_steps:
        with st.expander("Проміжні етапи: детектор → фільтр → верифікація MIP"):
            for s in rep["sentences"]:
                st.markdown(f"**{s['id']}.** {s['text']}")
                st.markdown(f"Детектор: `{s['detector_raw'] or '—'}`  \n"
                            f"Кандидати після фільтра службових слів: `{', '.join(s['candidates']) or '—'}`  \n"
                            f"Підтверджено: `{', '.join(s['verified']) or '—'}`")
                if s["verification"]:
                    st.dataframe(pd.DataFrame(s["verification"]).rename(columns={
                        "word": "слово", "contextual_meaning": "контекстне значення",
                        "basic_meaning": "базове значення", "is_metaphor": "метафора"}),
                        width="stretch", hide_index=True)
                if s.get("error"):
                    st.caption(f"Помилка: {s['error']}")

    d1, d2 = st.columns(2)
    d1.download_button("Завантажити JSON", json.dumps(rep, ensure_ascii=False, indent=1),
                       file_name="metaphor_report.json", mime="application/json")
    d2.download_button("Завантажити CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="metaphors.csv", mime="text/csv", disabled=df.empty)
