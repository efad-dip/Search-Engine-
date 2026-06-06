# Career Search Engine

A specialized search engine built from scratch for the domain of career recommendation based on academic courses and personal interests. The system implements a custom inverted index with TF-IDF ranking, boolean query support, and a web-based user interface deployed on Hugging Face Spaces.

---

## Live Demo

**Hugging Face Space:** [https://huggingface.co/spaces/Efad1096/career-search-engine]

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Preprocessing](#preprocessing)
- [Inverted Index](#inverted-index)
- [Query Processing](#query-processing)
- [Ranking](#ranking)
- [User Interface](#user-interface)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Allowed Libraries](#allowed-libraries)
- [Sample Queries](#sample-queries)
- [Authors](#authors)

---

## Project Overview

This project implements a specialized search engine targeting the career recommendation domain. Given a dataset of academic courses mapped to career options and personal interest areas, the system allows users to search using natural language queries and retrieves the most relevant career paths ranked by relevance score.

The core of the system is a hand-built inverted index. No pre-built search or indexing libraries such as Elasticsearch or Whoosh are used anywhere in the pipeline.

---

## Dataset

**File:** `CareerRecommenderDataset.csv`

**Domain:** Career recommendation based on academic background and personal interests

**Structure:**
- `Courses` — Name of the academic course or program
- `Career_Options` — Comma-separated list of associated career paths
- Interest columns — Binary (Yes/No) columns representing hobbies and skills such as coding, music, biology, drawing, travelling, and others

**Document construction:** Each unique course in the dataset is treated as one searchable document. The text of each document is composed of the course name, its career options, and all interest columns where the value is Yes. This produces a rich, searchable text representation per document.

---

## System Architecture

```
Raw CSV Dataset
      |
      v
  build_documents()
      |
      v
  preprocess()          <-- NLTK tokenization, stopword removal, stemming
      |
      v
  InvertedIndex.build() <-- term -> { doc_id -> [positions] }
      |
      v
  QueryEngine.search()  <-- boolean retrieval + TF-IDF scoring
      |
      v
  Ranked Results        <-- displayed in web UI or console
```

---

## Preprocessing

Each document goes through the following preprocessing pipeline before indexing:

1. **Lowercasing** — all text is converted to lowercase
2. **Special character removal** — non-alphanumeric characters are replaced with whitespace
3. **Tokenization** — text is split into tokens using NLTK word_tokenize
4. **Stopword removal** — common English stopwords are removed using the NLTK stopwords corpus
5. **Stemming** — tokens are reduced to their root form using the NLTK PorterStemmer

These steps ensure that queries such as "coding" and "coder" resolve to the same index term, and that common words like "the" or "and" do not affect ranking.

---

## Inverted Index

The inverted index is implemented entirely from scratch using Python dictionaries and the `collections.defaultdict` structure.

**Structure:**

```
{
  "term" : {
    doc_id_1 : [position_1, position_2, ...],
    doc_id_2 : [position_1, ...],
    ...
  },
  ...
}
```

Each term maps to a dictionary of document IDs, where each document ID maps to the list of positions at which that term appears in the document. This allows the system to compute term frequency directly from the position list length, and supports future extension to phrase queries.

**Additional data stored:**
- `doc_lengths` — total token count per document, used for TF normalization
- `num_docs` — total number of indexed documents, used for IDF computation

---

## Query Processing

The query engine supports three query modes:

**1. AND query**
The user explicitly writes AND between terms (case-insensitive). The system computes the intersection of the posting lists for all terms. Only documents containing every query term are returned.

Example: `music AND dancing`

**2. OR query**
The user explicitly writes OR between terms. The system computes the union of the posting lists. Any document containing at least one query term is returned.

Example: `biology OR chemistry`

**3. Multi-word query (default)**
When no boolean operator is detected, the query is treated as an OR query automatically. All terms are looked up and the union of their posting lists is returned. Results are ranked by TF-IDF score.

Example: `coding mathematics`

In all modes, results are ranked by their aggregated TF-IDF score before being returned to the user.

---

## Ranking

The system uses TF-IDF (Term Frequency - Inverse Document Frequency) to score and rank documents.

**Term Frequency (TF):**

```
TF(term, doc) = count of term in doc / total tokens in doc
```

**Inverse Document Frequency (IDF):**

```
IDF(term) = log((N + 1) / (df + 1)) + 1

where N  = total number of documents
      df = number of documents containing the term
```

**TF-IDF Score:**

```
TF-IDF(term, doc) = TF(term, doc) * IDF(term)
```

**Document Score:**

For a multi-term query, the final score for a document is the sum of TF-IDF scores across all query terms:

```
score(doc, query) = sum of TF-IDF(term, doc) for each term in query
```

Documents are sorted by this score in descending order and the top K results are returned.

---

## User Interface

The web interface is deployed as a Hugging Face Space using Gradio. It is styled with a dark theme and provides the following features:

- Search bar with Enter key support
- Search mode selector: Auto-detect, AND, or OR
- Adjustable result count (1 to 20)
- Ranked result cards showing course name, career chips, and matched interest tags
- Keyword highlighting in all result fields
- TF-IDF relevance score displayed per result
- Live index statistics bar showing document count, vocabulary size, and average token length
- CSV upload panel to re-index a new dataset without restarting

The Colab notebook (used for development and demonstration) contains all backend steps without the UI cell, which is handled exclusively by the Hugging Face deployment.

---

## Installation

**Requirements:**

```
gradio>=4.44.0
pandas>=2.0.0
numpy>=1.24.0
nltk>=3.8.0
```

**Install dependencies:**

```bash
pip install gradio pandas numpy nltk
```

**Run locally:**

```bash
python app.py
```

---

## Usage

**Google Colab (development and demo):**

1. Open `Career_Search_Engine.ipynb` in Google Colab
2. Run cells 1 through 5 in order
3. Upload `CareerRecommenderDataset.csv` when prompted in cell 2
4. Use cell 7 to run test queries in the console
5. Use cell 8 to view index statistics

**Hugging Face (web interface):**

1. Visit the live demo link at the top of this README
2. Upload `CareerRecommenderDataset.csv` using the upload panel at the bottom of the page
3. Type any interest, skill, or hobby into the search bar
4. Press Search or hit Enter

---

## Project Structure

```
career-search-engine/
|
|-- app.py                          # Hugging Face Space application
|-- requirements.txt                # Python dependencies
|-- README.md                       # This file
|-- Career_Search_Engine.ipynb      # Google Colab development notebook
|-- CareerRecommenderDataset.csv    # Dataset (upload separately)
```

---

## Allowed Libraries

| Library   | Purpose                                      |
|-----------|----------------------------------------------|
| NumPy     | Numerical operations and statistics          |
| Pandas    | Dataset loading and grouping                 |
| NLTK      | Tokenization, stopword removal, stemming     |
| Gradio    | Web interface for Hugging Face deployment    |

No search or indexing libraries (Elasticsearch, Whoosh, Solr, Lucene, etc.) are used anywhere in this project.

---

## Sample Queries

| Query                        | Mode     | Description                                      |
|------------------------------|----------|--------------------------------------------------|
| `coding`                     | Single   | Careers related to coding                        |
| `coding mathematics`         | OR auto  | Careers matching coding or mathematics           |
| `music AND dancing`          | AND      | Careers that require both music and dancing      |
| `biology OR chemistry`       | OR       | Careers in biology or chemistry                  |
| `travelling photography`     | OR auto  | Careers matching travel or photography interests |
| `economics accounting`       | OR auto  | Finance and business-related careers             |
| `sports cricket OR football` | OR       | Sports-related career paths                      |

---
