import logfire
from threading import Lock
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS
from app.guardrails.search import ApiEmbeddingsIndex


_rails: LLMRails | None = None
_guard_lock = Lock()


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses openai/gpt-oss-120b for fast intent classification at the gate —
    the heavier llama-3.3-70b-versatile is reserved for the RAG pipeline.
    """
    global _rails

    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required for guardrails API embeddings")
    # The search provider reads the key directly, keeping it out of config logs.

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-120b",
        temperature=0,
        timeout=30,
        max_retries=0,
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    # Explicitly select remote embeddings. NeMo's default FastEmbed model
    # downloads ONNX weights on first use and can SIGILL on the Render host.
    for model in config.models:
        if model.type == "embeddings":
            model.parameters = {
                "http_options": {"timeout": 30000},
            }

    _rails = LLMRails(config, llm=guard_llm)
    _rails.register_embedding_search_provider("api_cosine", ApiEmbeddingsIndex)
    logfire.info(" NeMo Guardrails initialised openai/gpt-oss-120b.")
    
    


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        raise RuntimeError("Guardrails are not initialized")

    with logfire.span("Guardrails Check"):
        # NeMo lazily builds shared indexes; avoid concurrent first-use builds.
        if not _guard_lock.acquire(timeout=5):
            raise RuntimeError("Guardrails are busy. Please retry shortly.")
        try:
            result = _rails.generate(
                messages=[{"role": "user", "content": message}],
                options={"log": {"internal_events": True}},
            )
        finally:
            _guard_lock.release()

        events = result.log.internal_events if result.log else None
        if any(
            event.get("type") == "InternalSystemActionFinished"
            and event.get("is_success") is False
            for event in events or []
        ):
            raise RuntimeError("Guardrail processing failed; check the server logs")
        response = result.response
        if isinstance(response, list):
            response = response[0] if response else {}

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = response.get("content", "") if isinstance(response, dict) else str(response)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("Guardrails passed.")
        return False, None
