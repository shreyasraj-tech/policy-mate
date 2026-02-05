try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from langchain.retrievers import BM25Retriever

try:
    from langchain_community.retrievers import EnsembleRetriever
except ImportError:
    from langchain.retrievers import EnsembleRetriever

from langchain_community.vectorstores import DocArrayInMemorySearch
from typing import List
from langchain_core.documents import Document

def get_hybrid_retriever(vector_store: DocArrayInMemorySearch, chunks: List[Document], k: int = 5):
    """
    Creates a hybrid retriever combining a dense retriever (DocArrayInMemorySearch with cosine similarity) and a sparse retriever (BM25).

    Args:
        vector_store (DocArrayInMemorySearch): The in-memory vector store for dense retrieval with cosine similarity.
        chunks (List[Document]): The list of documents for BM25.
        k (int): The number of documents to retrieve from each retriever.

    Returns:
        EnsembleRetriever: The hybrid retriever.
    """
    if not chunks:
        raise ValueError("Chunks cannot be empty for BM25 retriever.")

    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # Initialize dense retriever using Maximal Marginal Relevance (MMR) when available
    try:
        # Preferred: LangChain MaxMarginalRelevanceRetriever
        from langchain.retrievers import MaxMarginalRelevanceRetriever

        vector_retriever = MaxMarginalRelevanceRetriever.from_retriever(
            vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k * 4}),
            fetch_k=k * 4,
            k=k,
            lambda_mult=0.7,
        )
    except Exception:
        try:
            # Some vectorstores support mmr directly via as_retriever
            vector_retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k * 4, "lambda_mult": 0.7}
            )
        except Exception:
            # Fallback to standard similarity retriever
            vector_retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )

    # Initialize Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]  # Can be tuned - equal weight to sparse and dense retrieval
    )

    print(f"🔗 Hybrid retriever initialized:")
    print(f"   📊 BM25 (sparse) weight: 0.5")
    print(f"   🧠 In-memory vector store (dense, cosine similarity) weight: 0.5")
    print(f"   🔢 Retrieval count (k): {k}")

    return ensemble_retriever
