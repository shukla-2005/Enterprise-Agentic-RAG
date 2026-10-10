# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.

import logfire
import os
import logging
from contextlib import asynccontextmanager
from threading import Thread
from dotenv import load_dotenv

load_dotenv()
try:
    logfire.configure(
        token=os.getenv("LOGFIRE_TOKEN"),
        send_to_logfire="if-token-present",
    )
except Exception:
    logging.exception("Logfire configuration failed; continuing without tracing")

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, HTTPException, Response

from pydantic import BaseModel
from typing import Optional


rag_agent = None
guard = None


def initialize_backend():
    # Keep SDK imports and model downloads off the server's startup path.
    global rag_agent, guard
    from app.agents.graph import rag_agent as agent
    from app.guardrails import initialize_rails, guard as guard_query

    initialize_rails()
    rag_agent = agent
    guard = guard_query


def _initialize_backend(application):
    try:
        initialize_backend()
    except Exception:
        logging.exception("Backend initialization failed")
        application.state.backend_status = "error"
    else:
        application.state.backend_status = "ready"


@asynccontextmanager
async def lifespan(application):
    application.state.backend_status = "starting"
    Thread(target=_initialize_backend, args=(application,), daemon=True).start()
    yield


app = FastAPI(title="Enterprise Agentic RAG API", lifespan=lifespan)


def require_ready():
    status = getattr(app.state, "backend_status", "starting")
    if status != "ready":
        detail = (
            "Backend is starting. Please try again shortly."
            if status == "starting"
            else "Backend initialization failed. Check the backend server logs."
        )
        raise HTTPException(status_code=503, detail=detail)


@app.get("/health")
def health(response: Response):
    status = getattr(app.state, "backend_status", "starting")
    response.status_code = 200 if status == "ready" else 503
    return {"status": status}

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"
    
    
@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    require_ready()
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    require_ready()
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }
    
    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)
        
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        logfire.error(f"Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }
