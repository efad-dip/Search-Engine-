---
title: Career Search Engine
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🔍 Career Search Engine

A specialized search engine built from scratch using a **custom Inverted Index** and **TF-IDF ranking** — no Elasticsearch, no Whoosh, no pre-built search libraries.

## Features
- ✅ Custom Inverted Index: `term → { doc_id → [positions] }`
- ✅ TF-IDF ranking
- ✅ Boolean AND / OR query support
- ✅ Multi-word queries with automatic mode detection
- ✅ Keyword highlighting in results
- ✅ NLTK preprocessing (tokenization, stopword removal, stemming)

## How to Use
1. Upload your `CareerRecommenderDataset.csv` using the upload panel at the bottom
2. Type any interest or hobby into the search bar
3. Press **Search** or hit **Enter**

## Query Examples
| Query | Mode | Meaning |
|---|---|---|
| `coding mathematics` | OR (auto) | Find careers matching coding OR mathematics |
| `music AND dancing` | AND | Find careers requiring BOTH music AND dancing |
| `biology OR chemistry` | OR | Find careers in biology or chemistry |
| `travelling photography` | OR (auto) | Find careers matching travel or photography |

## Tech Stack
- **Language**: Python
- **Libraries**: NumPy, Pandas, NLTK, Gradio
- **Search**: Custom Inverted Index (no search libraries)
- **Ranking**: TF-IDF (Term Frequency × Inverse Document Frequency)
- **UI**: Gradio with custom dark theme
