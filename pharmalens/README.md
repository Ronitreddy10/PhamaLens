# PharmaLens

Local pharmaceutical product lifecycle knowledge base with hybrid retrieval, grounded Groq answers, citations, and section-level version-change detection.

## Setup

From the directory that contains the `pharmalens` folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r pharmalens/requirements.txt
cp pharmalens/.env.example .env
set -a; source .env; set +a
```

Add your Groq API key to `.env`. Embeddings run locally with `sentence-transformers/all-MiniLM-L6-v2`, so no OpenAI key is needed. The embedding model is downloaded on first ingestion.

## Run

The supplied six Semaglutide PDFs are in `pharmalens/data/documents/`. Additional PDFs can simply be dropped there.

```bash
python -m pharmalens.ingest.watcher
uvicorn pharmalens.api.app:app --reload --port 8000
streamlit run pharmalens/ui/app.py
```

Open <http://localhost:8501>. API documentation is at <http://localhost:8000/docs>.

## Test

```bash
python -m pytest pharmalens/tests
```

Document classification is entirely driven by `pharmalens/config/doc_types.yaml`. Add a taxonomy entry there and select an existing parser strategy; no Python change is needed.
