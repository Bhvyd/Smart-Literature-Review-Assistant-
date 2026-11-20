# Smart Literature Review Assistant

A web app that helps you review research papers on a topic.

You give it a topic (or your own PDFs). It collects the papers and lets you:

- **Papers**: search and rank them.
- **Map**: see which papers are similar.
- **Citations**: see who cites whom and which papers matter most.
- **Ask**: ask a question and get an answer that cites the papers.

The search engine (BM25, TF-IDF, cosine similarity, rank fusion) is written by hand in plain Python, in `rag_core/`. No search library is used there.

---

## 1. What you need

- **Python 3.11 or newer** (tested on 3.13 and 3.14). Get it from python.org.
- **Internet** the first time. The app downloads a language model (about 450 MB) once.
- **About 3 GB of free disk space.**
- **A Gemini API key** (free) for the Ask, Summary and Draft features. Get one at https://aistudio.google.com/apikey.
  Without a key, everything else still works.

You do **not** need Node.js. The web interface is already built (`web/dist`).

---

## 2. Set up (once)

Open a terminal in this folder.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

The install takes a few minutes (PyTorch is large).

Then open `.env` in a text editor and paste your key after `GEMINI_API_KEY=`.

---

## 3. First run

The app starts with an empty library. Click **Build a collection** and either:

- type a topic and let the app fetch papers (about 20 minutes for 100 papers), or
- upload your own PDFs.

Nothing is preloaded, so the first collection is yours.

---

## 4. Run

**Windows**

```powershell
.venv\Scripts\python -m uvicorn app.main:app
```

**macOS / Linux**

```bash
.venv/bin/python -m uvicorn app.main:app
```

Open **http://127.0.0.1:8000** in your browser. Press `Ctrl+C` in the terminal to stop.

The first search with Dense or Hybrid mode can take about 30 seconds while the model loads.

---

## 5. How to use it

1. **Pick a collection** on the left, or click **Build a collection**.
   - *Topic search*: type a topic. The app fetches papers from arXiv and OpenAlex.
     A new topic takes about 20 minutes for 100 papers. Progress shows on the left.
   - *Upload PDFs*: up to 25 PDFs (30 MB each, 200 MB in total).
   - A running build can be stopped with **Cancel build**. Any collection can be removed with **Delete**.
2. **Papers**: type a query and pick a mode.
   - *Lexical (BM25)*: matches exact words.
   - *Dense (SPECTER2)*: matches meaning.
   - *Hybrid*: combines both (best overall).
   Click a paper to see its details and a summary.
3. **Map**: each dot is a paper. Close dots are similar. Colours are clusters.
4. **Citations**: arrows point from the citing paper. The side list ranks papers by influence (PageRank and HITS).
5. **Ask**: ask a question. The answer cites its sources as [1], [2]. If the papers do not cover the question, it says so instead of guessing.
6. **Draft related work** and **Compare papers**: select papers (or a map cluster) to draft a paragraph or find agreements and conflicts. These are AI drafts. Check them before use.

---

## 6. How it works

```
Topic or PDFs
  -> fetch papers (arXiv, OpenAlex) and read the PDFs
  -> split the text into chunks
  -> turn papers and chunks into vectors (SPECTER2)
  -> store everything in SQLite
  -> search with the hand-written engine (rag_core)
  -> show papers, map, citation graph
  -> for Ask: send the best passages to Gemini, which must cite them
```

**The hand-written search engine (`rag_core/`)**

| File | What it does |
|---|---|
| `tokenize.py` | Splits text into words. |
| `index.py` | Builds the inverted index (which words appear in which chunk). |
| `lexical.py` | BM25 and TF-IDF cosine scoring. |
| `dense.py` | Cosine similarity between vectors. |
| `fuse.py` | Reciprocal Rank Fusion (combines BM25 and dense rankings). |
| `select.py` | Picks the top results in a fixed, repeatable order. |
| `prompt.py` | Builds the numbered passages sent to the language model. |
| `pipeline.py` | Ties it together: `Retriever.search(query, mode)`. |

Formulas used:

- BM25: `idf(t) * f*(k1+1) / (f + k1*(1 - b + b*len/avglen))`, with `k1 = 1.5`, `b = 0.75`.
- Cosine: `(q . d) / (|q| * |d|)`.
- Reciprocal Rank Fusion: `sum of 1 / (60 + rank)` over the two rankings.

A test (`tests/test_import_guard.py`) checks that `rag_core` uses only the Python standard library.

**Other folders**

| Folder | What it holds |
|---|---|
| `app/` | The web API (FastAPI). |
| `services/` | Features: search, map, graph, answers, summaries, drafts, comparisons, builds. |
| `ingest/` | Fetching from arXiv and OpenAlex, reading PDFs, chunking. |
| `embed/` | The SPECTER2 model wrapper. |
| `store/` | The SQLite database. |
| `evaluation/` | Measurement code (precision, recall, nDCG, latency, and more). |
| `web/` | The React interface. `web/dist` is the built version the server uses. |
| `tests/` | Automated tests. |
| `data/eval/` | Evaluation labels and the stored results. |

---

## 7. Results (measured on the retrieval augmented generation collection)

| What | Result |
|---|---|
| Search quality, nDCG@10 | Hybrid 0.782, BM25 0.755, TF-IDF 0.716, Dense 0.568 |
| Search speed (3,177 chunks) | BM25 9 ms, Hybrid 135 ms |
| Hand-written dense search vs FAISS | Same top 10 on 30 of 30 queries |
| Citation edges confirmed in the papers' own reference lists | 90% (252 of 279) |
| Answer sentences not supported by their cited source | 5% (2 of 40) |
| Summary vs real abstract | ROUGE-L 0.32, BERTScore 0.88 |

The relevance labels were made by AI raters, not people. Treat these numbers as a guide.

---

## 8. Tests and measurements (optional)

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```

Measurements (they run against a collection already in your local database and write to `data/eval/`;
the id below is the one used for the numbers in section 7):

```powershell
.venv\Scripts\python -m cli eval latency   --collection 96a1e1be1aec4a87ba961a406a84db6d
.venv\Scripts\python -m cli eval retrieval --collection 96a1e1be1aec4a87ba961a406a84db6d --labels data/eval/labels.json
.venv\Scripts\python -m cli eval refusal   --collection 96a1e1be1aec4a87ba961a406a84db6d --labels data/eval/questions.json
```

Run `.venv\Scripts\python -m cli eval --help` for the full list.

To change the interface, install Node.js 20+, then run `npm install` and `npm run build` inside `web/`.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not found | Install Python from python.org and tick "Add to PATH". |
| Install fails on `torch` | Make sure Python is 64-bit and 3.11 or newer. |
| Ask says "no API key" | Put your key in `.env` after `GEMINI_API_KEY=`, then restart. |
| Ask says "quota exhausted" | The free Gemini tier allows about 500 requests a day. Wait, or use a paid key. |
| First dense search is slow | The model is loading. Later searches are fast. |
| A build seems stuck | Check the progress on the left. A new topic takes about 20 minutes. |
| Port 8000 is busy | Run with `--port 8001` and open http://127.0.0.1:8001. |

---

## 10. Limits

- Open-access papers only (arXiv). No paywalled journals.
- Works best for computer science, AI and physics topics.
- Answers and drafts come from a language model. Always check the cited source.
- It is a single-user tool that runs on your own computer.
