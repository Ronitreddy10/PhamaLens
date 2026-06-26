---
title: PharmaLens API
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# PharmaLens

PharmaLens is a local proof-of-concept knowledge system for pharmaceutical product lifecycle documents. It indexes regulatory PDFs, retrieves grounded evidence, and answers product questions with citations instead of relying on model memory.

The current POC focuses on semaglutide lifecycle documents, including EPAR product information, FDA PSG documents, extracted tables, figure captions/vision descriptions, and an evaluation benchmark.

## What it does

- Hybrid retrieval over pharmaceutical PDF chunks
- Synonym expansion for pharma abbreviations and clinical terms
- Modality tagging, so answers distinguish mandatory, recommended, optional, prohibited, and permitted language
- Product disambiguation for Ozempic, Wegovy, and Rybelsus-style queries
- Structured clinical trial table extraction for numerical answers
- Figure and table extraction with searchable metadata
- FastAPI backend with a static dashboard UI
- Evaluation suite for retrieval, faithfulness, citation accuracy, and modality preservation

## Repository layout

```text
pharmalens/
  agent/          Answer generation and KB agent logic
  api/            FastAPI app
  config/         Settings, document taxonomy, synonyms
  data/           Local documents, extracted tables, eval artifacts
  eval/           PharmaEval benchmark and scoring scripts
  ingest/         PDF parsing, chunking, tables, figures, embeddings
  retrieval/      Hybrid search, reranking, filters, structured table lookup
  tests/          Unit tests
  ui/             Static dashboard and legacy Streamlit app
vercel.json       Vercel static dashboard deployment config
Dockerfile        Hugging Face Spaces backend deployment config
```

## Setup

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r pharmalens/requirements.txt
cp pharmalens/.env.example .env
```

Add your API keys to `.env`. Do not commit `.env`.

Example:

```bash
GROQ_API_KEY=your_key_here
```

Embeddings run locally with Sentence Transformers, so an OpenAI embedding key is not required for the current setup.

## Run locally

Start the API:

```bash
source .venv/bin/activate
python -m uvicorn pharmalens.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open the dashboard:

```text
http://127.0.0.1:8000/static/dashboard.html
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Ingest documents

Place PDFs in:

```text
pharmalens/data/documents/
```

Then run:

```bash
python -m pharmalens.ingest.watcher
```

The watcher parses documents, chunks text, tags modality, expands synonyms, extracts tables/figures, and updates the local vector store.

## Run tests

```bash
python -m pytest pharmalens/tests
```

## Run evaluation

Retrieval and faithfulness evaluation:

```bash
python -m pharmalens.eval.run_eval --full
```

Ablation study:

```bash
python -m pharmalens.eval.ablation
```

Recent POC evaluation results are stored under:

```text
pharmalens/eval/results/
```

## Vercel deployment

The static dashboard is designed to deploy cleanly on Vercel.

After pushing this repo to GitHub:

1. Open Vercel.
2. Import the GitHub repo.
3. Use **Other** as the framework preset.
4. Keep the settings from `vercel.json`:
   - Build command: `node pharmalens/ui/write-config.js`
   - Output directory: `pharmalens/ui`

Important: Vercel is hosting the frontend dashboard only. The chat/API features need the FastAPI backend deployed separately.

If the backend is deployed somewhere else, add this Vercel environment variable:

```bash
PHARMALENS_API_URL=https://your-backend-url.example.com
```

Without that variable, the dashboard uses same-origin API calls, which works locally but not on a static-only Vercel site.

## Hugging Face Spaces backend deployment

The FastAPI backend can be deployed as a Hugging Face Docker Space. This is a better fit than Render free tier for this app because the RAG stack needs more than 512MB RAM.

After pushing to GitHub:

1. Open Hugging Face Spaces.
2. Create a new Space.
3. Choose **Docker** as the Space SDK.
4. Add the repo files to the Space, or push this repo to the Space git remote.
5. Add this Space secret:

```bash
GROQ_API_KEY=your_key_here
```

The Docker Space exposes the API on port `7860` and starts:

```bash
python -m uvicorn pharmalens.api.app:app --host 0.0.0.0 --port 7860
```

On first startup, the API auto-ingests the bundled PDFs into a local Chroma vector store if the store is empty. If the PDFs are not present in the Space repository, it downloads them from the GitHub repo first. This avoids Hugging Face's plain-git binary-file rejection while keeping the deployed API self-contained.

After Hugging Face gives you a backend URL, add it to Vercel:

```bash
PHARMALENS_API_URL=https://your-space-name.hf.space
```

Then redeploy the Vercel frontend.

## GitHub push note

If GitHub already has an initial README commit and `git push` says `fetch first`, run:

```bash
git pull --allow-unrelated-histories origin main
git push -u origin main
```

## Disclaimer

PharmaLens is a research/education POC. It is not medical advice, regulatory advice, or a validated clinical decision system. Always verify answers against the source documents.
