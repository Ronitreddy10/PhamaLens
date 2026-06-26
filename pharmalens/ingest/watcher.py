"""Watch a directory and run classify → chunk → embed for each new PDF."""

import time
from pathlib import Path

import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from pharmalens.ingest.chunker import chunk_document
from pharmalens.ingest.classifier import classify
from pharmalens.ingest.embedder import is_already_indexed, upsert_chunks
from pharmalens.paths import CONFIG_DIR, resolve_data_path

with (CONFIG_DIR / "settings.yaml").open() as handle:
    SETTINGS = yaml.safe_load(handle)
WATCH_DIR = resolve_data_path(SETTINGS["ingest"]["watch_dir"])
SUPPORTED_EXTENSIONS = set(SETTINGS["ingest"]["supported_extensions"])


def ingest_file(filepath: str) -> dict:
    result = {"filepath": filepath, "status": "error", "chunks": 0, "doc_id": None, "delta_alerts": []}
    try:
        metadata = classify(filepath)
        result["doc_id"] = metadata.doc_id
        if is_already_indexed(metadata.doc_id):
            result["status"] = "skipped"
            return result
        chunks = chunk_document(filepath, metadata)
        if not chunks:
            result["status"] = "empty"
            return result
        result["chunks"] = upsert_chunks(chunks)
        result["status"] = "success"
        try:
            from pharmalens.agent.delta_detector import detect_changes
            result["delta_alerts"] = detect_changes(metadata)
        except Exception as exc:
            result["delta_error"] = str(exc)
    except Exception as exc:
        result["error"] = str(exc)
        print(f"[watcher] ERROR ingesting {filepath}: {exc}")
    return result


def ingest_directory(directory: str | Path = WATCH_DIR) -> list[dict]:
    return [ingest_file(str(path)) for path in sorted(Path(directory).rglob("*.pdf"))]


class PharmaLensEventHandler(FileSystemEventHandler):
    def _handle(self, filepath: str) -> None:
        if Path(filepath).suffix.lower() in SUPPORTED_EXTENSIONS:
            time.sleep(1)
            ingest_file(filepath)

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event) -> None:
        if not event.is_directory:
            self._handle(event.dest_path)


def start_watcher(watch_dir: str | None = None) -> None:
    directory = Path(watch_dir) if watch_dir else WATCH_DIR
    directory.mkdir(parents=True, exist_ok=True)
    ingest_directory(directory)
    observer = Observer()
    observer.schedule(PharmaLensEventHandler(), str(directory), recursive=True)
    observer.start()
    print(f"[watcher] Watching {directory}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    start_watcher()
