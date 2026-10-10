import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072

_active_model = None
_model_type: str | None = None


# Model initialisation 

def _probe_gemini():
    """Verify the collection's embedding model; never switch vector spaces."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probe")
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
        return model
    except Exception as e:
        raise RuntimeError("Gemini embeddings unavailable; cannot query the existing vector collection") from e


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"


# Public helpers 

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM


# Batch embedding with retry 

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        # Backoff: 5s -> 10s -> 20s -> 40s -> 60s to accommodate per-minute quota resets
        for attempt in range(6):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < 5:
                    wait = min(60, 5 * (2 ** attempt))
                    logfire.warning(
                        f"Gemini rate limit hit -- retrying in {wait}s "
                        f"(attempt {attempt + 1}/6)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 6 attempts.")
    raise RuntimeError("Embedding model is not initialized")


# Public API (same signatures as before)

def embed_query(query: str) -> list[float]:
    _init()
    return _active_model.embed_query(query)


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
