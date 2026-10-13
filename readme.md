# HireFlow: AI-Powered Recruitment Platform

HireFlow searches a pool of resumes and ranks candidates for a job using **hybrid search**
(BM25 keywords + FAISS vector similarity), Google Gemini for query expansion and candidate
evaluation, and a Streamlit web interface. Everything runs locally except the Gemini API calls.

## Features
- **Hybrid search:** BM25 for exact keywords plus FAISS (local vector index) for meaning
- **Smart routing:** short queries use fast vector search, long ones use multi-query + hybrid + AI re-rank
- **Filters:** required skills, location, minimum years of experience
- **AI evaluation:** fit score 0-100 with strengths, gaps and risks per candidate (rule-based fallback if Gemini is unavailable)
- **Resume parsing:** extracts skills, location, title and experience from PDFs
- **Memory:** logs searches and viewed candidates
- **Evaluation:** RAGAS metrics, with a heuristic fallback

## Setup
1. Python 3.11 or 3.12
2. Create and activate a virtual environment:
```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # Mac / Linux
```
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your free key from https://aistudio.google.com/apikey
5. Put your resume PDFs in `data/resumes/`

## Usage
```
python main.py index                                   # build the search indexes
python main.py search "senior accountant" --mode deep --rerank
python main.py stats
streamlit run ui/app.py                                # web app
```

## Project structure
```
main.py            command-line interface
core/              ingestion, parsing, FAISS vector store, hybrid indexer, filters,
                   re-ranker, search router, memory, evaluator, pipeline
utils/             config, helpers, schemas, embeddings, multi-query generator
ui/app.py          Streamlit app
data/              resumes and generated indexes (not committed)
```

## Notes
- Your API key lives in `.env`, which is git-ignored. Never commit it.
- Indexes are rebuilt with `python main.py index`, and embeddings are cached.
- If a Gemini model name stops working, change `LLM_MODEL` in `.env`.