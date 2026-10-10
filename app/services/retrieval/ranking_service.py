import time
import logfire


def rerank_documents(
    query: str,
    documents: list[str],
    top_n: int = 5
) -> list[str]:
    """
    Production-safe reranking fallback.

    FlashRank/ONNX is disabled on Render because the native
    ONNX runtime can crash with:
        Illegal instruction (core dumped)

    Documents are returned in the original Qdrant similarity order.
    """

    if not documents:
        return []

    start_time = time.time()

    logfire.info(
        "[Reranker] FlashRank disabled in production. "
        "Using Qdrant similarity ranking."
    )

    results = documents[:top_n]

    duration = time.time() - start_time

    logfire.info(
        f"[Reranker] Returned {len(results)} documents "
        f"in {duration:.4f}s using Qdrant ranking."
    )

    return results