"""
RAG (Retrieval-Augmented Generation) engine.

Usa ChromaDB como vector store local con embeddings LOCALES (all-MiniLM-L6-v2
via ONNX) — sin costo de API, corre 100% en tu Mac.

Workflow:
    1. Ingest: add_document(text, metadata) → embedded y guardado localmente.
    2. Query:  retrieve(query_text, k) → top-k passages concatenados.
    3. El LLM evaluador usa el contexto recuperado para mejorar su estimación.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Nombre de colección separado de la versión anterior (OpenAI embeddings)
# para evitar conflictos de dimensión de vectores.
_COLLECTION_NAME = "polymarket_rag_local"


class RagEngine:
    """
    Wrapper async sobre ChromaDB con embeddings locales.

    Los embeddings se generan con all-MiniLM-L6-v2 via ONNX (ya instalado
    como dependencia de chromadb) — no requiere ninguna API key.
    """

    def __init__(self, chroma_path: Path) -> None:
        self._path       = chroma_path
        self._collection = None
        self._ready      = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Crea o carga la colección ChromaDB con embeddings locales."""
        await asyncio.to_thread(self._sync_init)
        log.info("RAG engine listo (chroma_path=%s, colección=%s)", self._path, _COLLECTION_NAME)

    def _sync_init(self) -> None:
        import chromadb

        # DefaultEmbeddingFunction usa all-MiniLM-L6-v2 via ONNX — gratis y local
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            ef = DefaultEmbeddingFunction()
        except Exception as exc:
            log.warning("No se pudo cargar DefaultEmbeddingFunction: %s — usando embedding básico.", exc)
            ef = None

        client = chromadb.PersistentClient(path=str(self._path))

        kwargs: dict[str, Any] = {
            "name": _COLLECTION_NAME,
            "metadata": {"hnsw:space": "cosine"},
        }
        if ef is not None:
            kwargs["embedding_function"] = ef

        self._collection = client.get_or_create_collection(**kwargs)
        self._ready = True

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def add_document(
        self,
        text: str,
        source: str = "manual",
        title: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Embed y guarda un documento. Retorna el hash del contenido (ID estable).
        Silenciosamente salta si el contenido exacto ya fue ingerido.
        """
        if not self._ready:
            raise RuntimeError("Llama initialize() antes de add_document().")

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        await asyncio.to_thread(
            self._sync_add, text, content_hash, source, title, extra_metadata or {}
        )
        return content_hash

    def _sync_add(
        self,
        text: str,
        content_hash: str,
        source: str,
        title: str,
        extra_metadata: dict[str, Any],
    ) -> None:
        # Evitar duplicados por hash
        existing = self._collection.get(ids=[content_hash])
        if existing and existing["ids"]:
            log.debug("Documento ya ingerido (hash=%s).", content_hash[:12])
            return

        metadata = {
            "source":       source,
            "title":        title,
            "content_hash": content_hash,
            **extra_metadata,
        }
        self._collection.add(
            documents=[text],
            ids=[content_hash],
            metadatas=[metadata],
        )
        log.debug("Ingerido: '%s' (hash=%s).", title or source, content_hash[:12])

    async def add_documents_bulk(self, items: list[dict[str, Any]]) -> list[str]:
        """Ingerir múltiples documentos. Cada ítem debe tener: text, source, title."""
        hashes = []
        for item in items:
            h = await self.add_document(
                text=item.get("text", ""),
                source=item.get("source", "bulk"),
                title=item.get("title", ""),
                extra_metadata=item.get("metadata", {}),
            )
            hashes.append(h)
        return hashes

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def retrieve(self, query: str, top_k: int = 5) -> str:
        """
        Recupera los pasajes más relevantes para `query`.
        Retorna el texto concatenado listo para inyectar en el prompt.
        """
        if not self._ready or not query.strip():
            return ""
        return await asyncio.to_thread(self._sync_retrieve, query, top_k)

    def _sync_retrieve(self, query: str, top_k: int) -> str:
        count = self._collection.count()
        if count == 0:
            return ""

        n = min(top_k, count)
        results = self._collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        snippets = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
            title = meta.get("title") or meta.get("source", f"Doc {i}")
            score_str = f" (similitud={1 - dist:.3f})"
            snippets.append(f"[{i}] {title}{score_str}\n{doc}")

        return "\n\n---\n\n".join(snippets)

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def document_count(self) -> int:
        """Número de documentos en la colección."""
        if not self._ready or self._collection is None:
            return 0
        return await asyncio.to_thread(lambda: self._collection.count())

    async def reset(self) -> None:
        """Vacía toda la colección (usar con cuidado)."""
        if self._collection is not None:
            await asyncio.to_thread(
                self._collection.delete, where={"source": {"$ne": "__never__"}}
            )
            log.warning("Colección RAG vaciada.")
