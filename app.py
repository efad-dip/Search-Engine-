"""
Career Search Engine — Hugging Face Space
Inverted Index + TF-IDF Ranking | No Elasticsearch / Whoosh used
"""

import re
import math
import time
import io
import os
import pandas as pd
import numpy as np
import gradio as gr
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# ── NLTK downloads ─────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PREPROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess(text: str) -> list[str]:
    """Lowercase → strip special chars → tokenize → stopword removal → stem."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    return [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 1]


def build_documents(df: pd.DataFrame) -> list[dict]:
    """
    Convert dataset rows into rich searchable documents.
    Each unique course → one document containing:
      course name + career options + all interests where value == 'Yes'
    """
    interest_cols = [c for c in df.columns if c not in ["Courses", "Career_Options"]]
    documents = []
    doc_id = 0

    for course, group in df.groupby("Courses"):
        career_options = group["Career_Options"].iloc[0]

        yes_interests = []
        for col in interest_cols:
            if (group[col] == "Yes").any():
                yes_interests.append(col.replace("_", " "))

        full_text = f"{course} {career_options} {' '.join(yes_interests)}"

        documents.append({
            "doc_id": doc_id,
            "course": course,
            "careers": career_options,
            "interests": yes_interests,
            "full_text": full_text,
        })
        doc_id += 1

    return documents


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INVERTED INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InvertedIndex:
    """
    Custom Inverted Index.
    Structure: { term → { doc_id → [position_list] } }
    Computes TF, IDF, and TF-IDF without any search library.
    """

    def __init__(self):
        self.index: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        self.doc_lengths: dict[int, int] = {}
        self.num_docs: int = 0
        self.documents: list[dict] = []

    def build(self, documents: list[dict]):
        self.documents = documents
        self.num_docs = len(documents)
        for doc in documents:
            tokens = preprocess(doc["full_text"])
            self.doc_lengths[doc["doc_id"]] = len(tokens)
            for pos, term in enumerate(tokens):
                self.index[term][doc["doc_id"]].append(pos)

    def get_postings(self, term: str) -> set[int]:
        stemmed = stemmer.stem(term.lower())
        return set(self.index.get(stemmed, {}).keys())

    def tf(self, term: str, doc_id: int) -> float:
        stemmed = stemmer.stem(term.lower())
        count = len(self.index.get(stemmed, {}).get(doc_id, []))
        length = self.doc_lengths.get(doc_id, 1)
        return count / length if length > 0 else 0.0

    def idf(self, term: str) -> float:
        stemmed = stemmer.stem(term.lower())
        df = len(self.index.get(stemmed, {}))
        return math.log((self.num_docs + 1) / (df + 1)) + 1.0

    def tf_idf(self, term: str, doc_id: int) -> float:
        return self.tf(term, doc_id) * self.idf(term)

    def boolean_and(self, terms: list[str]) -> set[int]:
        if not terms:
            return set()
        result = self.get_postings(terms[0])
        for t in terms[1:]:
            result &= self.get_postings(t)
        return result

    def boolean_or(self, terms: list[str]) -> set[int]:
        result: set[int] = set()
        for t in terms:
            result |= self.get_postings(t)
        return result

    @property
    def vocab_size(self) -> int:
        return len(self.index)

    def top_terms(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted([(t, len(p)) for t, p in self.index.items()], key=lambda x: -x[1])[:n]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  QUERY ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QueryEngine:
    """
    Parses user queries (AND / OR / multi-word),
    performs boolean retrieval, and ranks with TF-IDF.
    """

    def __init__(self, inv_index: InvertedIndex):
        self.idx = inv_index

    def parse_query(self, raw: str) -> tuple[str, list[str]]:
        raw = raw.strip()
        if re.search(r"\bAND\b", raw, re.IGNORECASE):
            terms = [p.strip() for p in re.split(r"\bAND\b", raw, flags=re.IGNORECASE) if p.strip()]
            return "AND", terms
        elif re.search(r"\bOR\b", raw, re.IGNORECASE):
            terms = [p.strip() for p in re.split(r"\bOR\b", raw, flags=re.IGNORECASE) if p.strip()]
            return "OR", terms
        else:
            return "OR", raw.split()

    def score(self, doc_id: int, terms: list[str]) -> float:
        return sum(self.idx.tf_idf(t, doc_id) for t in terms)

    def search(self, raw_query: str, top_k: int = 10) -> list[dict]:
        if not raw_query.strip():
            return []
        mode, terms = self.parse_query(raw_query)
        candidates = self.idx.boolean_and(terms) if mode == "AND" else self.idx.boolean_or(terms)
        if not candidates:
            return []
        results = []
        for doc_id in candidates:
            doc = self.idx.documents[doc_id]
            results.append({
                "doc_id": doc_id,
                "score": round(self.score(doc_id, terms), 5),
                "course": doc["course"],
                "careers": doc["careers"],
                "interests": doc["interests"],
                "mode": mode,
                "query_terms": terms,
            })
        results.sort(key=lambda x: -x["score"])
        return results[:top_k]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GLOBAL ENGINE (loaded once at startup)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

inv_index = InvertedIndex()
engine: QueryEngine | None = None
INDEX_STATS = {}

def load_engine(csv_path: str = "CareerRecommenderDataset.csv"):
    global inv_index, engine, INDEX_STATS
    if not os.path.exists(csv_path):
        return False
    df = pd.read_csv(csv_path)
    documents = build_documents(df)
    inv_index = InvertedIndex()
    inv_index.build(documents)
    engine = QueryEngine(inv_index)
    INDEX_STATS = {
        "docs": inv_index.num_docs,
        "terms": inv_index.vocab_size,
        "avg_len": round(sum(inv_index.doc_lengths.values()) / max(inv_index.num_docs, 1), 1),
    }
    return True

# Try loading from bundled CSV at startup
_loaded = load_engine()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UI HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACCENT_COLORS = ["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#C77DFF"]

def highlight(text: str, terms: list[str]) -> str:
    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(
            lambda m: f'<mark class="hl">{m.group()}</mark>', text
        )
    return text


def interest_tags(interests: list[str], terms: list[str], max_show: int = 8) -> str:
    shown = interests[:max_show]
    html = ""
    for interest in shown:
        matched = any(t.lower() in interest.lower() for t in terms)
        cls = "tag tag-match" if matched else "tag"
        html += f'<span class="{cls}">{interest}</span>'
    if len(interests) > max_show:
        html += f'<span class="tag tag-more">+{len(interests) - max_show} more</span>'
    return html


def score_ring(score: float, rank: int) -> str:
    """SVG circular score indicator."""
    pct = min(score * 800, 100)
    radius = 20
    circumference = 2 * math.pi * radius
    stroke_dash = (pct / 100) * circumference
    color = ACCENT_COLORS[rank % len(ACCENT_COLORS)]
    return f"""
    <svg width="56" height="56" viewBox="0 0 56 56">
      <circle cx="28" cy="28" r="{radius}" fill="none" stroke="#1e1e2e" stroke-width="5"/>
      <circle cx="28" cy="28" r="{radius}" fill="none" stroke="{color}" stroke-width="5"
        stroke-dasharray="{stroke_dash:.1f} {circumference:.1f}"
        stroke-linecap="round" transform="rotate(-90 28 28)"/>
      <text x="28" y="33" text-anchor="middle" font-size="10"
        font-family="'DM Mono', monospace" fill="{color}" font-weight="700">
        #{rank+1}
      </text>
    </svg>"""


def render_results(results: list[dict], query: str, elapsed_ms: float) -> str:
    if not results:
        return f"""
        <div class="empty-state">
          <div class="empty-icon">◎</div>
          <h3>No results for &ldquo;{query}&rdquo;</h3>
          <p>Try broader terms, or switch from AND → OR mode.</p>
          <div class="suggestions">
            <span class="sug-label">Try:</span>
            <span class="sug">coding mathematics</span>
            <span class="sug">music AND dancing</span>
            <span class="sug">biology doctor</span>
          </div>
        </div>"""

    mode = results[0]["mode"]
    terms = results[0]["query_terms"]
    mode_badge = f'<span class="mode-badge mode-{mode.lower()}">{mode}</span>'

    header = f"""
    <div class="results-header">
      <span class="result-count">
        <span class="count-num">{len(results)}</span> result{"s" if len(results) != 1 else ""}
      </span>
      <span class="meta-sep">·</span>
      {mode_badge}
      <span class="meta-sep">·</span>
      <span class="elapsed">{elapsed_ms}ms</span>
      <span class="meta-sep">·</span>
      <span class="tfidf-label">ranked by TF-IDF</span>
    </div>"""

    cards = ""
    for i, r in enumerate(results):
        course_hl = highlight(r["course"], terms)
        careers_hl = highlight(r["careers"], terms)
        tags_html = interest_tags(r["interests"], terms)
        ring = score_ring(r["score"], i)
        is_top = "card card-top" if i == 0 else "card"

        # Career chips
        career_list = [c.strip() for c in r["careers"].split(",")]
        career_chips = "".join(
            f'<span class="career-chip">{highlight(c, terms)}</span>'
            for c in career_list[:4]
        )
        if len(career_list) > 4:
            career_chips += f'<span class="career-chip chip-more">+{len(career_list)-4}</span>'

        cards += f"""
        <div class="{is_top}" style="animation-delay:{i*0.06}s">
          <div class="card-left">
            {ring}
          </div>
          <div class="card-body">
            <div class="card-course">{course_hl}</div>
            <div class="career-chips">{career_chips}</div>
            <div class="card-interests">{tags_html}</div>
          </div>
          <div class="card-score">
            <span class="score-val">{r['score']:.4f}</span>
            <span class="score-lbl">score</span>
          </div>
        </div>"""

    return header + f'<div class="cards-grid">{cards}</div>'


def render_stats() -> str:
    if not INDEX_STATS:
        return "<div class='stats-bar'>Upload a CSV to begin</div>"
    return f"""
    <div class='stats-bar'>
      <div class='stat'><span class='stat-num'>{INDEX_STATS['docs']}</span><span class='stat-lbl'>Documents</span></div>
      <div class='stat-div'></div>
      <div class='stat'><span class='stat-num'>{INDEX_STATS['terms']:,}</span><span class='stat-lbl'>Index Terms</span></div>
      <div class='stat-div'></div>
      <div class='stat'><span class='stat-num'>{INDEX_STATS['avg_len']}</span><span class='stat-lbl'>Avg Tokens/Doc</span></div>
      <div class='stat-div'></div>
      <div class='stat'><span class='stat-num'>TF-IDF</span><span class='stat-lbl'>Ranking</span></div>
    </div>"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GRADIO CALLBACKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def upload_csv(file):
    global engine
    if file is None:
        return "<div class='upload-msg upload-err'>❌ No file received.</div>", render_stats()
    try:
        df = pd.read_csv(file.name)
        documents = build_documents(df)
        inv_index.build(documents)
        engine_obj = QueryEngine(inv_index)
        global engine, INDEX_STATS
        engine = engine_obj
        INDEX_STATS = {
            "docs": inv_index.num_docs,
            "terms": inv_index.vocab_size,
            "avg_len": round(sum(inv_index.doc_lengths.values()) / max(inv_index.num_docs, 1), 1),
        }
        return (
            f"<div class='upload-msg upload-ok'>✓ Indexed {inv_index.num_docs} documents · {inv_index.vocab_size:,} terms</div>",
            render_stats()
        )
    except Exception as e:
        return f"<div class='upload-msg upload-err'>❌ Error: {e}</div>", render_stats()


def do_search(query: str, mode_choice: str, top_k: int):
    if engine is None:
        return "<div class='upload-prompt'>⬆ Upload your CareerRecommenderDataset.csv first</div>"
    if not query.strip():
        return "<div class='empty-prompt'>Type something in the search bar above ↑</div>"

    # Override mode
    words = query.strip().split()
    if mode_choice == "AND — all terms must match":
        query_str = " AND ".join(words)
    elif mode_choice == "OR — any term matches":
        query_str = " OR ".join(words)
    else:
        query_str = query.strip()

    t0 = time.perf_counter()
    results = engine.search(query_str, top_k=int(top_k))
    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    return render_results(results, query, elapsed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STYLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=Instrument+Sans:wght@400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #0a0a12;
  --surface:   #11111d;
  --surface2:  #181828;
  --border:    #2a2a40;
  --text:      #e8e8f0;
  --muted:     #6b6b8a;
  --accent1:   #FF6B6B;
  --accent2:   #FFD93D;
  --accent3:   #6BCB77;
  --accent4:   #4D96FF;
  --accent5:   #C77DFF;
  --font-head: 'Syne', sans-serif;
  --font-body: 'Instrument Sans', sans-serif;
  --font-mono: 'DM Mono', monospace;
  --radius:    12px;
  --glow:      0 0 30px rgba(77,150,255,0.12);
}

.gradio-container {
  background: var(--bg) !important;
  min-height: 100vh;
  font-family: var(--font-body) !important;
  color: var(--text) !important;
}

/* remove Gradio chrome */
footer { display: none !important; }
.gr-prose, .prose { color: var(--text) !important; }
.svelte-1gfkn6j { background: transparent !important; }

/* ── Hero Header ── */
#hero {
  text-align: center;
  padding: 52px 20px 32px;
  position: relative;
  overflow: hidden;
}
#hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 60% 50% at 50% -10%, rgba(77,150,255,0.18) 0%, transparent 70%),
    radial-gradient(ellipse 40% 30% at 20% 100%, rgba(199,125,255,0.1) 0%, transparent 60%),
    radial-gradient(ellipse 40% 30% at 80% 100%, rgba(107,203,119,0.08) 0%, transparent 60%);
  pointer-events: none;
}

.hero-eyebrow {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 3px;
  color: var(--accent4);
  text-transform: uppercase;
  margin-bottom: 16px;
  opacity: 0.9;
}

.hero-title {
  font-family: var(--font-head);
  font-size: clamp(38px, 6vw, 72px);
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: -2px;
  margin-bottom: 16px;
  background: linear-gradient(135deg,
    var(--accent4) 0%,
    #9bb5ff 30%,
    var(--accent5) 60%,
    var(--accent1) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-sub {
  font-family: var(--font-body);
  font-size: 15px;
  color: var(--muted);
  max-width: 480px;
  margin: 0 auto 28px;
  line-height: 1.6;
}

/* ── Stats Bar ── */
.stats-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 50px;
  padding: 10px 28px;
  width: fit-content;
  margin: 0 auto 40px;
  font-family: var(--font-mono);
  font-size: 12px;
}
.stat { display: flex; flex-direction: column; align-items: center; padding: 0 18px; }
.stat-num { color: var(--accent4); font-weight: 700; font-size: 14px; }
.stat-lbl { color: var(--muted); font-size: 10px; margin-top: 1px; letter-spacing: 0.5px; }
.stat-div { width: 1px; height: 28px; background: var(--border); }

/* ── Search Container ── */
#search-wrap {
  max-width: 780px;
  margin: 0 auto;
  padding: 0 20px;
}

/* Gradio textbox overrides */
#search-wrap textarea,
#search-wrap input[type="text"] {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 50px !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
  font-size: 16px !important;
  padding: 16px 28px !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
  outline: none !important;
}
#search-wrap textarea:focus,
#search-wrap input[type="text"]:focus {
  border-color: var(--accent4) !important;
  box-shadow: 0 0 0 3px rgba(77,150,255,0.15) !important;
}

/* ── Controls Row ── */
#controls { max-width: 780px; margin: 16px auto 0; padding: 0 20px; }
#controls label { color: var(--muted) !important; font-size: 12px !important; font-family: var(--font-mono) !important; }
#controls select, #controls .gr-dropdown {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
}
#controls input[type="range"] { accent-color: var(--accent4) !important; }

/* Search Button */
#search-btn button {
  background: linear-gradient(135deg, var(--accent4), var(--accent5)) !important;
  border: none !important;
  border-radius: 50px !important;
  color: #fff !important;
  font-family: var(--font-head) !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  padding: 12px 32px !important;
  letter-spacing: 0.5px !important;
  transition: transform 0.15s, box-shadow 0.15s !important;
  box-shadow: 0 4px 20px rgba(77,150,255,0.3) !important;
}
#search-btn button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 28px rgba(77,150,255,0.45) !important;
}

/* Example queries */
.examples-row {
  max-width: 780px;
  margin: 16px auto 0;
  padding: 0 20px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.ex-label {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 1px;
  text-transform: uppercase;
}
.ex-chip {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 4px 14px;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--accent4);
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
}
.ex-chip:hover {
  background: rgba(77,150,255,0.12);
  border-color: var(--accent4);
}

/* ── Divider ── */
.divider {
  max-width: 780px;
  margin: 36px auto;
  border: none;
  border-top: 1px solid var(--border);
}

/* ── Upload Section ── */
.upload-section {
  max-width: 780px;
  margin: 0 auto;
  padding: 0 20px 40px;
}
.upload-section label { color: var(--muted) !important; font-size: 12px !important; font-family: var(--font-mono) !important; }
.upload-msg { font-family: var(--font-mono); font-size: 13px; padding: 10px 16px; border-radius: 8px; margin-top: 8px; }
.upload-ok  { background: rgba(107,203,119,0.12); border: 1px solid var(--accent3); color: var(--accent3); }
.upload-err { background: rgba(255,107,107,0.12); border: 1px solid var(--accent1); color: var(--accent1); }
.upload-prompt, .empty-prompt {
  font-family: var(--font-mono);
  color: var(--muted);
  text-align: center;
  padding: 60px 20px;
  font-size: 14px;
}

/* ── Results ── */
#results-area { max-width: 780px; margin: 0 auto; padding: 0 20px 60px; }

.results-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 20px;
  flex-wrap: wrap;
}
.count-num { color: var(--text); font-weight: 700; font-size: 15px; }
.meta-sep { color: var(--border); }
.elapsed { color: var(--accent3); }
.tfidf-label { color: var(--accent5); }
.mode-badge {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  padding: 2px 10px;
  border-radius: 20px;
}
.mode-and { background: rgba(255,107,107,0.15); color: var(--accent1); border: 1px solid var(--accent1); }
.mode-or  { background: rgba(77,150,255,0.15); color: var(--accent4); border: 1px solid var(--accent4); }

/* Cards */
.cards-grid { display: flex; flex-direction: column; gap: 14px; }

.card {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
  animation: slide-in 0.35s ease both;
}
.card:hover {
  border-color: var(--accent4);
  box-shadow: var(--glow);
  transform: translateX(4px);
}
.card-top {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  background: linear-gradient(135deg, #0e0e20, #131328);
  border: 1px solid rgba(77,150,255,0.35);
  border-radius: var(--radius);
  padding: 20px 22px;
  box-shadow: 0 0 24px rgba(77,150,255,0.1), inset 0 1px 0 rgba(255,255,255,0.03);
  transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
  animation: slide-in 0.35s ease both;
}
.card-top:hover {
  border-color: var(--accent4);
  box-shadow: 0 0 40px rgba(77,150,255,0.2);
  transform: translateX(4px);
}

@keyframes slide-in {
  from { opacity: 0; transform: translateX(-16px); }
  to   { opacity: 1; transform: translateX(0); }
}

.card-left { flex-shrink: 0; }
.card-body { flex: 1; min-width: 0; }

.card-course {
  font-family: var(--font-head);
  font-size: 17px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 8px;
  line-height: 1.3;
}

.career-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 10px;
}
.career-chip {
  background: rgba(77,150,255,0.1);
  border: 1px solid rgba(77,150,255,0.25);
  border-radius: 20px;
  padding: 3px 12px;
  font-family: var(--font-body);
  font-size: 12px;
  color: var(--accent4);
}
.chip-more {
  background: rgba(107,203,119,0.08);
  border-color: rgba(107,203,119,0.2);
  color: var(--accent3);
}

.card-interests {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}
.tag {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 2px 10px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--muted);
}
.tag-match {
  background: rgba(255,217,61,0.1);
  border-color: rgba(255,217,61,0.3);
  color: var(--accent2);
}
.tag-more {
  background: transparent;
  border: 1px dashed var(--border);
  color: var(--muted);
}

.card-score {
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
  padding-top: 4px;
}
.score-val {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 500;
  color: var(--accent3);
}
.score-lbl {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--muted);
  letter-spacing: 0.5px;
}

/* Highlight */
mark.hl {
  background: rgba(255,217,61,0.2);
  color: var(--accent2);
  border-radius: 3px;
  padding: 0 2px;
  font-weight: 600;
}

/* Empty State */
.empty-state {
  text-align: center;
  padding: 60px 20px;
  font-family: var(--font-body);
}
.empty-icon {
  font-size: 48px;
  color: var(--muted);
  margin-bottom: 16px;
  opacity: 0.4;
}
.empty-state h3 { font-family: var(--font-head); font-size: 22px; color: var(--text); margin-bottom: 8px; }
.empty-state p  { color: var(--muted); font-size: 14px; margin-bottom: 20px; }
.suggestions { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.sug-label { color: var(--muted); font-family: var(--font-mono); font-size: 12px; align-self: center; }
.sug {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 4px 14px;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--accent4);
}

/* Upload area */
.upload-area {
  border: 1.5px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--surface) !important;
  transition: border-color 0.2s !important;
}
.upload-area:hover { border-color: var(--accent4) !important; }

/* Responsive */
@media (max-width: 600px) {
  .card, .card-top { flex-direction: column; }
  .card-score { flex-direction: row; align-items: center; gap: 6px; }
  .hero-title { font-size: 36px; }
  .stats-bar { flex-wrap: wrap; border-radius: 12px; }
}
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GRADIO APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLES = [
    "coding mathematics",
    "music AND dancing",
    "biology doctor",
    "travelling photography",
    "economics accounting",
    "drawing AND painting",
    "chess puzzles",
    "sports cricket OR football",
]

with gr.Blocks(css=CSS, title="Career Search Engine") as demo:

    # ── Hero ──────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="hero">
      <div class="hero-eyebrow">◈ Inverted Index · TF-IDF Ranking</div>
      <h1 class="hero-title">Career Search Engine</h1>
      <p class="hero-sub">
        Discover career paths by searching your interests, hobbies, and skills.
        Powered by a custom-built inverted index with TF-IDF ranking.
      </p>
    </div>
    """)

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats_html = gr.HTML(value=render_stats())

    # ── Search Bar ────────────────────────────────────────────────────────────
    with gr.Column(elem_id="search-wrap"):
        query_input = gr.Textbox(
            placeholder='Search interests & careers...  e.g. "coding mathematics"  or  "music AND dancing"',
            label="",
            lines=1,
            show_label=False,
        )

    # ── Controls ──────────────────────────────────────────────────────────────
    with gr.Row(elem_id="controls"):
        mode_dd = gr.Dropdown(
            choices=["Auto-detect from query", "AND — all terms must match", "OR — any term matches"],
            value="Auto-detect from query",
            label="Search Mode",
            scale=3,
        )
        top_k_sl = gr.Slider(minimum=1, maximum=20, value=8, step=1, label="Max Results", scale=2)
        search_btn = gr.Button("Search ↵", variant="primary", scale=1, elem_id="search-btn")

    # ── Example chips ─────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="examples-row">
      <span class="ex-label">Try →</span>
      {"".join(f'<span class="ex-chip" onclick="document.querySelector(\'#search-wrap textarea\').value=\'{e}\';document.querySelector(\'#search-btn button\').click()">{e}</span>' for e in EXAMPLES[:6])}
    </div>
    """)

    gr.HTML('<hr class="divider">')

    # ── Results ───────────────────────────────────────────────────────────────
    with gr.Column(elem_id="results-area"):
        results_out = gr.HTML(
            value="<div class='empty-prompt'>Enter a query above and press Search or ↵ Enter</div>"
        )

    gr.HTML('<hr class="divider">')

    # ── Upload ────────────────────────────────────────────────────────────────
    with gr.Column(elem_classes="upload-section"):
        gr.HTML("""
        <div style='font-family:var(--font-mono,monospace);font-size:11px;
          letter-spacing:2px;text-transform:uppercase;color:#6b6b8a;margin-bottom:10px;'>
          ◈ Upload Dataset
        </div>""")
        csv_upload = gr.File(
            label="Upload CareerRecommenderDataset.csv",
            file_types=[".csv"],
            elem_classes="upload-area",
        )
        upload_msg = gr.HTML()

    # ── Wire events ───────────────────────────────────────────────────────────
    def _search(q, mode, k):
        return do_search(q, mode, k)

    search_btn.click(_search, [query_input, mode_dd, top_k_sl], results_out)
    query_input.submit(_search, [query_input, mode_dd, top_k_sl], results_out)
    csv_upload.change(upload_csv, csv_upload, [upload_msg, stats_html])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    demo.launch()
