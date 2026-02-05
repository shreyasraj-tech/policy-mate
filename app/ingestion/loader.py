"""Document ingestion and vector store helpers for Policy Mate.

Provides:
 - PDF text extraction (lazy imports)
 - Chunking using `utils.splitter.semantic_split` (defaults tuned for policy docs)
 - Cached embeddings and vector store creation
 - MMR retriever helper (diverse retrieval)

Functions are designed for programmatic use; module has no side-effects on import.
"""

from functools import lru_cache
from typing import List, Optional
import os
import requests
import logging

from langchain_core.documents import Document as LangChainDocument

from utils.splitter import semantic_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_pdf_text(url: str, timeout: int = 15) -> str:
    """Download and extract text from a PDF URL using PyMuPDF (lazy import).

    Args:
        url: Public URL to the PDF document.
        timeout: HTTP timeout in seconds.

    Returns:
        Full extracted text as a single string.
    """
    try:
        import fitz  # PyMuPDF (lazy)
    except Exception:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF extraction")

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    doc = fitz.open(stream=r.content, filetype="pdf")
    pages = []
    for p in doc:
        pages.append(p.get_text())
    doc.close()
    return "\n".join(pages)


def split_text_to_documents(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[LangChainDocument]:
    """Split text into a list of LangChain `Document` objects using the
    policy-aware `semantic_split`.

    Args:
        text: Input document text.
        chunk_size: Target chunk size (characters; ~tokens).
        chunk_overlap: Overlap length between chunks.

    Returns:
        List of `langchain_core.documents.Document` instances.
    """
    chunks = semantic_split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = [LangChainDocument(page_content=chunk) for chunk in chunks]
    return docs


@lru_cache(maxsize=4)
def get_embeddings_provider(provider: str = "cohere", model: Optional[str] = None):
    """Return a cached embeddings provider instance.

    Currently supports `cohere`, `ollama`, and `openai` (fallback).
    """
    if provider == "cohere":
        try:
            from langchain_cohere import CohereEmbeddings
            model = model or os.getenv("COHERE_MODEL", "embed-v4.0")
            return CohereEmbeddings(model=model)
        except Exception:
            logger.warning("CohereEmbeddings not available; falling back")

    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
            model = model or "all-minilm:33m"
            return OllamaEmbeddings(model=model)
        except Exception:
            logger.warning("OllamaEmbeddings not available; falling back")

    # Default: OpenAI-compatible embeddings
    try:
        from langchain_openai import OpenAIEmbeddings
        model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    except Exception:
        raise RuntimeError("No available embeddings provider (install Cohere, Ollama, or OpenAI embeddings)")


@lru_cache(maxsize=4)
def get_vectorstore_from_index(index_name: str, namespace: Optional[str], embeddings_provider) -> object:
    """Load an existing Pinecone/compatible vector store pointing at an index.

    Returns a vector store instance (implementation-opaque).
    """
    try:
        from langchain_pinecone import PineconeVectorStore
        return PineconeVectorStore.from_existing_index(
            embedding=embeddings_provider,
            index_name=index_name,
            namespace=namespace,
        )
    except Exception:
        # Fallback: raise descriptive error
        raise RuntimeError("PineconeVectorStore not available or index not found")


def mmr_select(docs, query_embedding, doc_embeddings, k: int = 5) -> List[LangChainDocument]:
    """Perform a simple greedy Maximal Marginal Relevance selection.

    Args:
        docs: list of Document objects (candidates)
        query_embedding: embedding vector of the query
        doc_embeddings: list of embedding vectors corresponding to docs
        k: number of documents to select

    Returns:
        Selected subset of `docs` (ordered)
    """
    import numpy as np

    def cosine(a, b):
        a = np.array(a)
        b = np.array(b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    selected = []
    if not docs:
        return []

    # Precompute similarities
    sim_to_query = [cosine(query_embedding, emb) for emb in doc_embeddings]
    remaining = set(range(len(docs)))

    # Pick highest scoring doc first
    first = int(max(remaining, key=lambda i: sim_to_query[i]))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < min(k, len(docs)) and remaining:
        best_score = None
        best_idx = None
        for idx in remaining:
            # compute marginal relevance = lambda * sim(query, doc) - (1-lambda) * max_sim_to_selected
            lambda_param = 0.7
            sim_q = sim_to_query[idx]
            max_sim_to_selected = max(cosine(doc_embeddings[idx], doc_embeddings[s]) for s in selected)
            mmr_score = lambda_param * sim_q - (1 - lambda_param) * max_sim_to_selected
            if best_score is None or mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [docs[i] for i in selected]


def mmr_retrieve(vector_store, embeddings_provider, query: str, k: int = 5, fetch_k: int = 20) -> List[LangChainDocument]:
    """Retrieve documents using MMR to promote diversity.

    Args:
        vector_store: Vector store instance with `similarity_search_with_score` method.
        embeddings_provider: Embeddings provider with `embed_query` method.
        query: Query string.
        k: Number of final results to return.
        fetch_k: Number of candidate documents to fetch before MMR selection.
    """
    # Step 1: fetch candidate documents with scores
    try:
        candidates = vector_store.similarity_search_with_score(query, k=fetch_k)
    except Exception:
        # fallback to simple similarity_search
        candidates = [(d, 0.0) for d in vector_store.similarity_search(query, k=fetch_k)]

    docs = [d for d, _ in candidates]

    if not docs:
        return []

    # Step 2: compute embeddings for query and docs (try to use provider methods)
    try:
        query_emb = embeddings_provider.embed_query(query)
    except Exception:
        # Try naming differences (cohere/OpenAI)
        try:
            query_emb = embeddings_provider.embed_documents([query])[0]
        except Exception:
            raise RuntimeError("Embeddings provider does not support embedding queries")

    doc_texts = [d.page_content for d in docs]
    try:
        doc_embs = embeddings_provider.embed_documents(doc_texts)
    except Exception:
        # If embedding whole docs fails, embed first 512 chars
        doc_embs = embeddings_provider.embed_documents([t[:1024] for t in doc_texts])

    selected = mmr_select(docs, query_emb, doc_embs, k=k)
    return selected



