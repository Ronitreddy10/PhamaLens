FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PHARMALENS_AUTO_INGEST=1 \
    PHARMALENS_DOWNLOAD_DEMO_DOCS=1 \
    PHARMALENS_LLM_MODEL=llama-3.1-8b-instant \
    PHARMALENS_MAX_COMPLETION_TOKENS=900 \
    PHARMALENS_MAX_CONTEXT_CHUNKS=4 \
    PHARMALENS_VISION_ENABLED=0 \
    PHARMALENS_CORS_ORIGINS=* \
    PHARMALENS_CORS_ORIGIN_REGEX=https://.*\\.vercel\\.app \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user pharmalens/requirements.txt ./pharmalens/requirements.txt

USER user

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r pharmalens/requirements.txt

COPY --chown=user . $HOME/app

EXPOSE 7860

CMD ["sh", "-c", "python -m uvicorn pharmalens.api.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
