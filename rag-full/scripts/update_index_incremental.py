"""Incrementally update the FAISS index from processed Markdown files."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.base_crawler import setup_logging
from src.indexer.build_vectorstore import (
    build_faiss_in_batches,
    create_embeddings,
    load_config,
    load_markdown_documents,
    resolve_project_path,
    save_vectorstore_atomically,
    split_documents,
)


LOGGER = logging.getLogger(__name__)


def _docstore_get(vectorstore: Any, doc_id: str):
    """Return a document from LangChain's in-memory docstore."""
    doc = vectorstore.docstore.search(doc_id)
    return None if isinstance(doc, str) else doc


def index_inventory(vectorstore: Any) -> dict[str, dict[str, Any]]:
    """Group current FAISS docstore entries by source_file."""
    inventory: dict[str, dict[str, Any]] = {}
    for doc_id in vectorstore.index_to_docstore_id.values():
        doc = _docstore_get(vectorstore, doc_id)
        if doc is None:
            continue
        source_file = doc.metadata.get("source_file")
        if not source_file:
            continue
        source_file = str(Path(source_file).resolve())
        entry = inventory.setdefault(
            source_file,
            {
                "content_hashes": set(),
                "docstore_ids": [],
                "chunks": 0,
                "source": doc.metadata.get("source"),
                "title": doc.metadata.get("title"),
            },
        )
        entry["docstore_ids"].append(doc_id)
        entry["chunks"] += 1
        if doc.metadata.get("content_hash"):
            entry["content_hashes"].add(doc.metadata.get("content_hash"))
    return inventory


def load_vectorstore(config: dict[str, Any], config_path: Path, index_dir: Path):
    """Load the existing FAISS index."""
    from langchain_community.vectorstores import FAISS

    if not (index_dir / "index.faiss").exists():
        return None
    embeddings = create_embeddings(config)
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _current_documents(markdown_dir: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    docs = load_markdown_documents(markdown_dir, config=config)
    return {str(Path(doc.metadata["source_file"]).resolve()): doc for doc in docs}


def _needs_update(indexed_entry: dict[str, Any] | None, doc: Any) -> bool:
    if indexed_entry is None:
        return True
    current_hash = doc.metadata.get("content_hash")
    return current_hash not in indexed_entry.get("content_hashes", set())


def _max_chunk_id(vectorstore: Any) -> int:
    max_id = -1
    for doc_id in vectorstore.index_to_docstore_id.values():
        doc = _docstore_get(vectorstore, doc_id)
        if doc is None:
            continue
        chunk_id = doc.metadata.get("chunk_id")
        if isinstance(chunk_id, int):
            max_id = max(max_id, chunk_id)
    return max_id


def _add_documents_in_batches(vectorstore: Any, chunks: list[Any], config: dict[str, Any]) -> None:
    batch_size = int(config.get("embedding", {}).get("index_batch_size", 512))
    started_at = time.monotonic()
    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunks[start:end]
        LOGGER.info("Embedding incremental chunks %d-%d / %d", start + 1, end, total)
        vectorstore.add_documents(batch)
        elapsed = time.monotonic() - started_at
        processed = end
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = (total - processed) / rate if rate > 0 else 0.0
        LOGGER.info(
            "Incremental indexed %d/%d chunks (%.1f%%), %.1f chunks/sec, eta %.1f min",
            processed,
            total,
            processed * 100 / total,
            rate,
            remaining / 60,
        )


def write_manifest(vectorstore: Any, index_dir: Path, config: dict[str, Any]) -> Path:
    """Write a compact manifest next to the FAISS files."""
    inventory = index_inventory(vectorstore)
    sources = []
    source_counts: dict[str, int] = defaultdict(int)
    for source_file, entry in sorted(inventory.items()):
        source = entry.get("source") or "unknown"
        source_counts[source] += int(entry["chunks"])
        sources.append(
            {
                "source_file": source_file,
                "source": source,
                "title": entry.get("title"),
                "content_hashes": sorted(entry["content_hashes"]),
                "chunks": entry["chunks"],
            }
        )
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding": config.get("embedding", {}),
        "total_chunks": len(vectorstore.index_to_docstore_id),
        "total_files": len(sources),
        "chunks_by_source": dict(sorted(source_counts.items())),
        "files": sources,
    }
    path = index_dir / "index_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def update_index(
    *,
    config_path: Path,
    dry_run: bool = False,
    delete_missing: bool = True,
) -> dict[str, Any]:
    """Apply an incremental FAISS update and return a summary."""
    config = load_config(config_path)
    project_config = config.get("project", {})
    markdown_dir = resolve_project_path(
        config_path,
        project_config.get("processed_markdown_dir", "data/processed/markdown"),
    )
    index_dir = resolve_project_path(config_path, project_config.get("index_dir", "data/index/faiss"))
    index_dir.mkdir(parents=True, exist_ok=True)

    current_docs = _current_documents(markdown_dir, config=config)
    vectorstore = load_vectorstore(config, config_path, index_dir)

    if vectorstore is None:
        LOGGER.info("No existing FAISS index found; building a new index from all Markdown")
        chunks = split_documents(list(current_docs.values()), config)
        if dry_run:
            return {
                "mode": "full_build",
                "files_to_add": len(current_docs),
                "chunks_to_add": len(chunks),
                "deleted_chunks": 0,
                "final_chunks": 0,
            }
        embeddings = create_embeddings(config)
        vectorstore = build_faiss_in_batches(chunks, embeddings, config)
        save_vectorstore_atomically(vectorstore, index_dir)
        manifest_path = write_manifest(vectorstore, index_dir, config)
        return {
            "mode": "full_build",
            "files_to_add": len(current_docs),
            "chunks_to_add": len(chunks),
            "deleted_chunks": 0,
            "final_chunks": len(vectorstore.index_to_docstore_id),
            "manifest": str(manifest_path),
        }

    inventory = index_inventory(vectorstore)
    files_to_add = [
        source_file
        for source_file, doc in current_docs.items()
        if _needs_update(inventory.get(source_file), doc)
    ]
    files_to_delete = []
    if delete_missing:
        files_to_delete.extend(source_file for source_file in inventory if source_file not in current_docs)
    files_to_delete.extend(source_file for source_file in files_to_add if source_file in inventory)
    files_to_delete = sorted(set(files_to_delete))

    docs_to_add = [current_docs[source_file] for source_file in sorted(files_to_add)]
    chunks_to_add = split_documents(docs_to_add, config) if docs_to_add else []
    next_chunk_id = _max_chunk_id(vectorstore) + 1
    for offset, chunk in enumerate(chunks_to_add):
        chunk.metadata["chunk_id"] = next_chunk_id + offset

    docstore_ids_to_delete = [
        doc_id
        for source_file in files_to_delete
        for doc_id in inventory[source_file]["docstore_ids"]
    ]
    summary = {
        "mode": "incremental",
        "indexed_files": len(inventory),
        "current_files": len(current_docs),
        "files_to_add": len(files_to_add),
        "files_to_delete": len(files_to_delete),
        "chunks_to_add": len(chunks_to_add),
        "chunks_to_delete": len(docstore_ids_to_delete),
        "initial_chunks": len(vectorstore.index_to_docstore_id),
    }
    LOGGER.info("Incremental index plan: %s", summary)

    if dry_run:
        return summary

    if docstore_ids_to_delete:
        LOGGER.info("Deleting %d stale chunks from FAISS", len(docstore_ids_to_delete))
        vectorstore.delete(ids=docstore_ids_to_delete)

    if chunks_to_add:
        _add_documents_in_batches(vectorstore, chunks_to_add, config)

    if docstore_ids_to_delete or chunks_to_add:
        save_vectorstore_atomically(vectorstore, index_dir)
    manifest_path = write_manifest(vectorstore, index_dir, config)
    summary["final_chunks"] = len(vectorstore.index_to_docstore_id)
    summary["manifest"] = str(manifest_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sources.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-delete-missing", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    setup_logging(PROJECT_ROOT / config.get("project", {}).get("logs_dir", "logs"))
    summary = update_index(
        config_path=config_path,
        dry_run=args.dry_run,
        delete_missing=not args.no_delete_missing,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

