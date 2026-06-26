"""FastAPI REST API for PharmaLens."""

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pharmalens.agent.kb_agent import query as kb_query
from pharmalens.ingest.watcher import ingest_file
from pharmalens.paths import DATA_DIR, PACKAGE_ROOT

app = FastAPI(title="PharmaLens", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(PACKAGE_ROOT / "ui")), name="static")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    conversation_history: list[dict] = Field(default_factory=list)
    metadata_filters: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    filepath: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "PharmaLens"}


@app.get("/")
def root():
    """Redirect root to the dashboard served by this API."""
    return RedirectResponse(url="/static/dashboard.html")


@app.post("/query")
def query_endpoint(request: QueryRequest):
    try:
        response = kb_query(request.query, request.conversation_history, request.metadata_filters)
        return {"answer": response.answer, "sources": [{
            "filename": chunk.metadata.get("filename"), "section": chunk.metadata.get("section_title", ""),
            "section_number": chunk.metadata.get("section_number", ""), "page": chunk.metadata.get("page_number"),
            "doc_type": chunk.metadata.get("doc_type"), "regulatory_body": chunk.metadata.get("regulatory_body"),
            "chunk_kind": chunk.metadata.get("chunk_kind", "text"),
            "dominant_modality": chunk.metadata.get("dominant_modality", "NEUTRAL"),
            "score": round(chunk.score, 4), "excerpt": chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""),
        } for chunk in response.sources], "delta_alerts": response.delta_alerts}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest", status_code=202)
def ingest_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    path = Path(request.filepath)
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF file not found")
    background_tasks.add_task(ingest_file, str(path))
    return {"status": "ingestion queued", "filepath": str(path)}


@app.get("/collection/stats")
def collection_stats():
    from pharmalens.ingest.embedder import get_collection
    return {"total_chunks": get_collection(with_embedding_function=False).count()}


@app.post("/eval/retrieval")
async def run_retrieval_eval_endpoint(background_tasks: BackgroundTasks, k: int = 5):
    import uuid
    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(_run_eval_task, job_id, k)
    return {"status": "started", "job_id": job_id, "message": f"Eval running with k={k}"}


def _run_eval_task(job_id: str, k: int):
    import json
    from pharmalens.eval.retrieval_evaluator import run_retrieval_eval
    try:
        results = run_retrieval_eval(k=k)
        path = DATA_DIR / "eval_results.json"
        path.write_text(json.dumps(results, indent=2))
        print(f"[eval] Job {job_id} complete. R@{k}: {results['overall']['recall_at_k']:.3f}")
    except Exception as exc:
        print(f"[eval] Job {job_id} failed: {exc}")


@app.get("/eval/results")
def get_eval_results():
    import json
    path = DATA_DIR / "eval_results.json"
    if not path.exists():
        return {"status": "no results yet — run POST /eval/retrieval first"}
    return json.loads(path.read_text())
