"""
ReviewMood: movie-review sentiment dashboard (Random Forest + TF-IDF).
Run locally:  streamlit run app.py
"""
import html
import json
from pathlib import Path

import joblib
import streamlit as st

from preprocess import clean_text

ART = Path("artifacts")
st.set_page_config(page_title="ReviewMood", page_icon="🎬", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
:root{--bg:#fff;--card:#FEF2EE;--tile:#FCE0D5;--track:#E3E3E8;--line:#EDEDF0;--ink:#2A2A33;--mute:#6B6B76;
--or:#FF5A1F;--or2:#FF8A5B;--pos:#22B455;--pos-bg:#DDF6E6;--neg:#E5484D;--neg-bg:#FDE3E3;}
html,body,[class*="css"],.stApp{font-family:'Poppins',sans-serif;color:var(--ink);}
.stApp{background:var(--bg);}
#MainMenu,footer{display:none!important;}
@media(min-width:577px){header[data-testid="stHeader"],[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarHeader"]{display:none!important;}}
header[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1240px;padding:1.4rem 2rem 4rem;}
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line);width:250px!important;min-width:250px!important;}
section[data-testid="stSidebar"]>div{padding:22px 16px;}

/* sidebar */
.brand{display:flex;align-items:center;gap:10px;margin:0 0 28px 6px;font-weight:700;font-size:21px;letter-spacing:.01em;color:var(--or);}
.mark{width:30px;height:30px;border-radius:9px 9px 15px 9px;background:var(--or);display:grid;place-items:center;color:#fff;font-size:15px;}
.grp{font-size:11px;color:var(--mute);margin:0 0 10px 8px;letter-spacing:.04em;}
.nav a{display:flex;align-items:center;gap:12px;padding:11px 14px;border-radius:9px;color:#55555F;text-decoration:none;font-size:14px;margin-bottom:6px;}
.nav a:hover{background:var(--card);} .nav a.on{background:var(--or);color:#fff;font-weight:500;}
.side-foot{position:fixed;bottom:20px;width:218px;border-top:1px solid var(--line);padding-top:16px;display:flex;align-items:center;gap:10px;}
.ava{width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,var(--or),#FFB08A);display:grid;place-items:center;font-size:18px;}
.side-foot b{display:block;font-size:13px;} .side-foot span{font-size:11.5px;color:var(--mute);}

/* top bar */
.top{font-size:20px;font-weight:500;padding:6px 0 18px;}
.stDownloadButton>button{background:#fff;border:1px solid var(--line);border-radius:10px;color:var(--ink);height:40px;font-size:13px;font-weight:500;font-family:'Poppins',sans-serif;}
.stDownloadButton>button:hover{border-color:var(--or);color:var(--or);}
.rule{height:1px;background:var(--line);margin:0 -2rem 22px;}

/* cards */
.card,.st-key-analyze{background:var(--card);border-radius:18px;padding:22px 24px;}
.card h3,.st-key-analyze h3{font-size:18px;font-weight:600;margin:0 0 14px;}
.gap{margin-top:18px;}
.kc{background:var(--card);border-radius:18px;padding:18px 22px;}
.kh{display:flex;align-items:center;gap:12px;font-size:15px;color:#3a3a44;}
.ic{width:38px;height:38px;border-radius:10px;background:var(--tile);color:var(--or);display:grid;place-items:center;}
.kv{display:flex;align-items:center;justify-content:space-between;margin:14px 0 12px;}
.kv b{font-size:34px;font-weight:600;letter-spacing:-.02em;}
.dl{font-size:11.5px;font-weight:500;border-radius:7px;padding:3px 9px;} .dl.up{background:var(--pos-bg);color:var(--pos);} .dl.nt{background:#ECECF0;color:var(--mute);}
.pb{height:11px;border-radius:99px;background:var(--track);overflow:hidden;} .pb i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#FFA57F,var(--or));}
.kcap{font-size:11.5px;margin-top:10px;color:#3a3a44;}

/* inputs */
.stTextArea textarea{border-radius:12px;border:1px solid #F0D5CA;background:#fff;color:var(--ink);font-size:14px;line-height:1.55;padding:14px;font-family:'Poppins',sans-serif;}
.stTextArea textarea:focus{border-color:var(--or);box-shadow:0 0 0 3px rgba(255,90,31,.14);}
div[data-baseweb="select"]>div{border-radius:10px;border-color:#F0D5CA;background:#fff;}
.stButton>button{background:var(--or);color:#fff;border:0;border-radius:8px;height:46px;padding:0 28px;font-weight:500;font-size:14px;font-family:'Poppins',sans-serif;}
.stButton>button:hover{background:#E8480F;color:#fff;} .stButton>button:focus-visible{outline:2px solid var(--ink);outline-offset:2px;}
.res .big{font-size:40px;font-weight:600;letter-spacing:-.02em;line-height:1.1;display:flex;align-items:center;justify-content:space-between;margin-top:8px;}
.res .lab{font-size:12px;font-weight:500;border-radius:7px;padding:3px 10px;}
.res .lab.p{background:var(--pos-bg);color:var(--pos);} .res .lab.n{background:var(--neg-bg);color:var(--neg);}
.res .pb{margin:12px 0;} .res p{font-size:13px;line-height:1.55;margin:0;color:#3a3a44;}
.hint{font-size:13px;color:var(--mute);background:#fff;border-radius:12px;padding:18px;margin-top:6px;}

/* mood grid */
.mg{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;}
.mc{aspect-ratio:1;border-radius:11px;display:grid;place-items:center;font-size:23px;background:#F6E4DD;}
.mc.p{background:var(--pos-bg);} .mc.n{background:var(--neg-bg);}
.avg{display:grid;place-items:center;padding:14px 0 12px;}
.avg .face{width:112px;height:112px;border-radius:50%;display:grid;place-items:center;font-size:68px;background:radial-gradient(circle at 35% 30%,#FFF1B8,#FFC83D);box-shadow:0 10px 24px rgba(255,160,40,.35);}
.avg p{font-size:12.5px;color:var(--mute);margin:14px 0 0;text-align:center;}
.scale{display:flex;gap:12px;justify-content:center;border-top:1px solid #F3D9CF;padding-top:14px;margin-top:6px;}
.scale span{font-size:22px;opacity:.35;} .scale span.on{opacity:1;transform:scale(1.2);}

/* top words */
.tp{background:var(--card);border-radius:18px;padding:18px 16px;text-align:center;margin-bottom:14px;}
.tp .av{width:56px;height:56px;border-radius:50%;margin:0 auto 10px;display:grid;place-items:center;font-size:28px;background:#fff;border:2px solid #FBD3C4;}
.tp b{display:block;font-size:15px;} .tp .rl{font-size:11.5px;color:var(--mute);margin:2px 0 10px;}
.tp.h{font-size:18px;font-weight:600;line-height:1.25;padding:20px;}

/* metrics + matrix */
.mr{display:grid;grid-template-columns:130px 1fr 52px;gap:14px;align-items:center;margin-bottom:13px;font-size:13px;}
.mr .vv{text-align:right;font-weight:600;}
.hm{display:grid;grid-template-columns:90px 1fr 1fr;gap:8px;}
.hm .h{font-size:11.5px;color:var(--mute);display:flex;align-items:center;justify-content:center;text-align:center;}
.hm .c{border-radius:12px;min-height:88px;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:25px;font-weight:600;}
.hm .c span{font-size:11px;font-weight:400;opacity:.85;}
.rw{display:grid;grid-template-columns:42px 1fr 120px;gap:14px;align-items:center;padding:12px 0;border-top:1px solid #F3D9CF;font-size:13.5px;}
.rw:first-of-type{border-top:0;}
.rw .fc{width:38px;height:38px;border-radius:10px;display:grid;place-items:center;font-size:20px;}
.rw .fc.p{background:var(--pos-bg);} .rw .fc.n{background:var(--neg-bg);}
.rw .tm{font-size:11.5px;color:var(--mute);text-align:right;}
@media(max-width:900px){.mr{grid-template-columns:100px 1fr 46px;}.rw{grid-template-columns:42px 1fr;}.rw .tm{display:none;}}
/* about */
.about{background:linear-gradient(135deg,#FF5A1F,#FF8A5B);border-radius:20px;padding:26px 30px;color:#fff;display:grid;grid-template-columns:1.25fr 1fr;gap:28px;margin:28px 0 0;}
.about h2{font-size:24px;font-weight:600;margin:0 0 10px;line-height:1.25;color:#fff;}
.about p{font-size:13.5px;line-height:1.65;margin:0 0 10px;color:rgba(255,255,255,.93);}
.about .nb{font-size:12px;color:rgba(255,255,255,.85);}
.steps{display:flex;flex-direction:column;gap:10px;}
.st1{background:rgba(255,255,255,.17);border-radius:14px;padding:12px 16px;display:flex;gap:14px;align-items:flex-start;}
.st1 i{font-style:normal;width:26px;height:26px;border-radius:50%;background:#fff;color:var(--or);font-weight:600;font-size:13px;display:grid;place-items:center;flex:none;}
.st1 b{display:block;font-size:13.5px;font-weight:600;} .st1 span{font-size:12px;opacity:.92;line-height:1.45;}
@media(max-width:900px){.about{grid-template-columns:1fr;}}
/* mobile */
.twg{display:grid;grid-template-columns:1fr;gap:14px;} .twg .tp{margin:0;}
@media(max-width:576px){
 section[data-testid="stSidebar"]{width:82vw!important;min-width:0!important;max-width:300px;}
 .side-foot{position:static;width:auto;margin-top:28px;}
 .block-container{padding:3.4rem 1rem 3rem;}
 .rule{margin:0 -1rem 16px;} .top{font-size:18px;padding-bottom:12px;}
 .about{padding:20px;gap:18px;} .about h2{font-size:20px;}
 .card,.st-key-analyze{padding:18px;border-radius:16px;} .card h3,.st-key-analyze h3{font-size:16px;}
 .kc{padding:16px 18px;} .kv b{font-size:28px;}
 .mg{gap:6px;} .mc{font-size:19px;border-radius:9px;}
 .avg .face{width:96px;height:96px;font-size:58px;}
 .res .big{font-size:28px;flex-wrap:wrap;gap:8px;}
 .mr{grid-template-columns:92px 1fr 42px;gap:10px;font-size:12px;}
 .hm{grid-template-columns:62px 1fr 1fr;gap:6px;} .hm .c{min-height:74px;font-size:21px;}
 .twg{grid-template-columns:1fr 1fr;gap:10px;} .tp{padding:14px 10px;} .tp b{font-size:14px;}
 .stButton>button,.stDownloadButton>button{width:100%;}
 .stTextArea textarea{font-size:16px;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ICONS = {
    "home": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/>',
    "edit": '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
    "grid": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="5"/>',
    "bar": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>',
}


def icon(name, size=18):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load(ART / "model.joblib")


@st.cache_data
def load_metrics():
    return json.loads((ART / "metrics.json").read_text())


if not (ART / "model.joblib").exists():
    st.error("Model file `artifacts/model.joblib` is missing. Run `python train.py` to create it.")
    st.stop()

bundle = load_model()
tfidf, rf, class_names = bundle["tfidf"], bundle["rf"], bundle["class_names"]
M = load_metrics()
rep = M["report"]
POS = class_names.index("positive")


def predict(text):
    """Return (label, p_positive) or None when the text has no usable words."""
    cleaned = clean_text(text)
    if not cleaned.strip():
        return None
    p = float(rf.predict_proba(tfidf.transform([cleaned]))[0][POS])
    return ("positive" if p >= 0.5 else "negative"), p


def mood(p):
    """Emoji + css class for a probability of 'positive'."""
    if p >= .85: return "😍", "p"
    if p >= .60: return "😊", "p"
    if p >= .40: return "😐", ""
    if p >= .15: return "🙁", "n"
    return "😡", "n"


SAMPLES = [
    "Absolutely loved this film! The acting is superb and the story kept me hooked.",
    "Good cast, but the pacing dragged and the ending felt rushed.",
    "Not worth your time. The plot is thin and the dialogue feels cheap.",
]
if "history" not in st.session_state:
    st.session_state.history = []
    for t in reversed(SAMPLES):
        label, p = predict(t)
        st.session_state.history.insert(0, {"text": t, "label": label, "p": p, "when": "example"})

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    nav = [("home", "Dashboard", "top", True), ("edit", "Analyze review", "analyze", False),
           ("grid", "Confusion matrix", "matrix", False), ("bar", "Performance", "perf", False),
           ("star", "Top words", "words", False), ("clock", "Recent", "recent", False), ("file", "About us", "about", False)]
    items = "".join(f'<a class="{"on" if on else ""}" href="#{a}">{icon(i)}{n}</a>' for i, n, a, on in nav)
    bp = M["search"]["best_params"]
    st.markdown(f"""<div class="brand"><div class="mark">🎬</div>REVIEWMOOD</div>
    <div class="grp">MANAGEMENT</div><div class="nav">{items}</div>
    <div class="side-foot"><div class="ava">🌲</div><div><b>Random Forest</b><span>{bp['n_estimators']} trees · IMDB 50K</span></div></div>""",
                unsafe_allow_html=True)

# ----------------------------------------------------------------- top bar
t1, t2 = st.columns([5, 1.2], vertical_alignment="center")
t1.markdown('<div class="top" id="top">Dashboard</div>', unsafe_allow_html=True)
t2.download_button("Export metrics", json.dumps(M, indent=2), "metrics.json", "application/json")
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

main, side = st.columns([3.3, 1], gap="medium")

# ================================================================= MAIN
with main:
    sup = [rep[c]["support"] for c in class_names]
    base = max(sup) / sum(sup)
    gain = (M["test_accuracy"] - base) * 100
    split = M["n_test"] / (M["n_test"] + M["n_train"])

    def kpi(ic, name, val, pill, pcls, pct, cap):
        return (f'<div class="kc"><div class="kh"><span class="ic">{icon(ic)}</span>{name}</div>'
                f'<div class="kv"><b>{val}</b><span class="dl {pcls}">{pill}</span></div>'
                f'<div class="pb"><i style="width:{pct*100:.1f}%"></i></div><div class="kcap">{cap}</div></div>')

    k1, k2, k3 = st.columns(3, gap="small")
    k1.markdown(kpi("bar", "Test accuracy", f"{M['test_accuracy']:.1%}", f"↑ {gain:.1f} pts", "up",
                    M["test_accuracy"], f"Compared to guessing one class ({base:.1%})"), unsafe_allow_html=True)
    k2.markdown(kpi("check", "Cross-validation F1", f"{M['search']['best_cv_f1_weighted']:.1%}", f"{M['search']['cv']}-fold", "nt",
                    M["search"]["best_cv_f1_weighted"], f"Tuned on {M['search']['tune_sample']:,} reviews"), unsafe_allow_html=True)
    k3.markdown(kpi("file", "Reviews tested", f"{M['n_test']:,}", f"{split:.0%} split", "nt",
                    split, f"Trained on {M['n_train']:,} reviews"), unsafe_allow_html=True)

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
    lc, rc = st.columns([1, 1.5], gap="small")

    # ---- left: review mood grid + average mood
    hist = st.session_state.history
    cells = ""
    for i in range(25):
        if i < len(hist):
            e, c = mood(hist[i]["p"]); cells += f'<div class="mc {c}">{e}</div>'
        else:
            cells += '<div class="mc"></div>'
    avg = sum(h["p"] for h in hist) / len(hist)
    ae, _ = mood(avg)
    scale = "".join(f'<span class="{"on" if e == ae else ""}">{e}</span>' for e in ["😡", "🙁", "😐", "😊", "😍"])
    lc.markdown(f"""<div class="card" id="recent-grid"><h3>Review Mood</h3><div class="mg">{cells}</div>
      <div style="font-size:11.5px;color:var(--mute);margin-top:12px">Latest 25 predictions, newest first.</div>
      <h3 style="margin-top:26px">Average Mood</h3>
      <div class="avg"><div class="face">{ae}</div><p>{avg:.0%} positive on average<br>across {len(hist)} analyzed reviews</p></div>
      <div class="scale">{scale}</div></div>""", unsafe_allow_html=True)

    # ---- right: analyzer + performance
    with rc:
        st.markdown('<div id="analyze"></div>', unsafe_allow_html=True)
        with st.container(border=False, key="analyze"):
            st.markdown('<h3>Analyze a review</h3>', unsafe_allow_html=True)
            examples = {"Positive example": "An absolutely brilliant film. The acting was superb, the story gripping, "
                                            "and the ending left me in tears. Highly recommended!",
                        "Negative example": "A complete waste of time. The plot made no sense, the acting was terrible "
                                            "and I was bored from start to finish."}
            choice = st.selectbox("Example", ["Start from an example"] + list(examples), label_visibility="collapsed")
            text = st.text_area("Review", value=examples.get(choice, ""), height=130,
                                placeholder="Write or paste a movie review…", label_visibility="collapsed")
            if st.button("Analyze review"):
                res = predict(text)
                if res is None:
                    st.markdown('<div class="hint">Add a few words about the movie, then select Analyze review.</div>',
                                unsafe_allow_html=True)
                else:
                    label, p = res
                    st.session_state.history.insert(0, {"text": text.strip(), "label": label, "p": p, "when": "just now"})
                    conf = max(p, 1 - p)
                    note = ("This review reads as positive. Enthusiastic wording like this usually signals a recommendation."
                            if label == "positive" else
                            "This review reads as negative. The wording points to disappointment with the film.")
                    st.markdown(f"""<div class="res"><div class="big"><span>{label.capitalize()} · {conf:.0%}</span>
                      <span class="lab {'p' if label == 'positive' else 'n'}">P(positive) {p:.0%}</span></div>
                      <div class="pb"><i style="width:{conf*100:.1f}%"></i></div><p>{note}</p></div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="hint">Your prediction appears here. Select an example or write a review, then analyze it.</div>',
                            unsafe_allow_html=True)

        def mbar(label, v):
            return (f'<div class="mr"><span>{label}</span><div class="pb"><i style="width:{v*100:.1f}%"></i></div>'
                    f'<span class="vv">{v:.0%}</span></div>')
        bars = "".join(mbar(f"{c.capitalize()} {m}", rep[c][k]) for c in ("positive", "negative")
                       for m, k in (("precision", "precision"), ("recall", "recall"), ("F1", "f1-score")))
        st.markdown(f'<div class="card gap" id="perf"><h3>Precision, recall and F1</h3>{bars}</div>', unsafe_allow_html=True)

    # ---- bottom row: confusion matrix + recent predictions
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
    b1, b2 = st.columns([1, 1.5], gap="small")
    (tn, fp), (fn, tp) = M["confusion_matrix"]
    big = max(tn, fp, fn, tp)

    def cell(v, lab):
        a = 0.10 + 0.90 * v / big
        return (f'<div class="c" style="background:rgba(255,90,31,{a:.2f});color:{"#fff" if a > .5 else "#7A3A22"}">'
                f'{v:,}<span>{lab}</span></div>')
    b1.markdown(f"""<div class="card" id="matrix"><h3>Confusion matrix</h3>
      <div class="hm"><div></div><div class="h">Predicted<br>negative</div><div class="h">Predicted<br>positive</div>
      <div class="h">Actual<br>negative</div>{cell(tn, "correct")}{cell(fp, "wrong")}
      <div class="h">Actual<br>positive</div>{cell(fn, "wrong")}{cell(tp, "correct")}</div></div>""", unsafe_allow_html=True)

    rows = ""
    for r in hist[:5]:
        e, c = mood(r["p"])
        txt = html.escape(r["text"][:95] + ("…" if len(r["text"]) > 95 else ""))
        rows += (f'<div class="rw"><div class="fc {c}">{e}</div><div>“{txt}”</div>'
                 f'<div class="tm">{r["label"].capitalize()} · {max(r["p"], 1 - r["p"]):.0%}<br>{r["when"]}</div></div>')
    b2.markdown(f'<div class="card" id="recent"><h3>Recent predictions</h3>{rows}</div>', unsafe_allow_html=True)

# ================================================================= SIDE: top words
with side:
    st.markdown('<div id="words"></div><div class="tp h">Top<br>Words</div>', unsafe_allow_html=True)
    feats = M["top_features"][:5]
    mx = feats[0]["importance"]
    cards = ""
    for f in feats:
        pos = f["leans"] == "positive"
        pill = ('<span class="dl up">' if pos else '<span class="dl" style="background:var(--neg-bg);color:var(--neg)">')
        cards += (f'<div class="tp"><div class="av">{"😊" if pos else "😞"}</div><b>{html.escape(f["feature"])}</b>'
                  f'<div class="rl">Leans {f["leans"]}</div>{pill}{"↑" if pos else "↓"} {f["importance"]/mx*100:.0f} score</span></div>')
    st.markdown(f'<div class="twg">{cards}</div>', unsafe_allow_html=True)

# ================================================================= ABOUT US (bottom)
st.markdown(f"""<div class="about" id="about"><div>
  <h2>About us: what is ReviewMood?</h2>
  <p>ReviewMood is a movie-review sentiment analyzer. Paste any English movie review and it tells you
  whether the writer felt <b>positive</b> or <b>negative</b>, with a confidence score.</p>
  <p>It was trained on the IMDB 50K reviews dataset ({M['eda']['n_rows_used']:,} unique reviews after removing duplicates)
  and tested on {M['n_test']:,} reviews it had never seen, where it is right about
  {M['test_accuracy']:.0%} of the time.</p>
  <div class="nb">Good to know: it reads written English film reviews. Sarcasm, mixed opinions and other
  topics (products, news) can fool it.</div></div>
  <div class="steps">
    <div class="st1"><i>1</i><div><b>Clean the text</b><span>Lowercase, remove punctuation, numbers and filler words, reduce words to their base form.</span></div></div>
    <div class="st1"><i>2</i><div><b>Turn words into numbers</b><span>TF-IDF scores the {M['tfidf_params']['max_features']:,} most useful words and word pairs.</span></div></div>
    <div class="st1"><i>3</i><div><b>Vote with {M['search']['best_params']['n_estimators']} decision trees</b><span>A Random Forest combines the trees' votes into one prediction.</span></div></div>
  </div></div>""", unsafe_allow_html=True)