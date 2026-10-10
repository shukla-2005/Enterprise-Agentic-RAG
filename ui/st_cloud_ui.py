import logging
import os
import uuid

import logfire
import requests
import streamlit as st

st.set_page_config(page_title="Enterprise Assistant", page_icon="✦", layout="wide")


def setting(name, default=None):
    try:
        return st.secrets.get(name) or os.getenv(name, default)
    except FileNotFoundError:
        return os.getenv(name, default)


@st.cache_resource
def configure_tracing():
    try:
        token = setting("LOGFIRE_TOKEN")
        if token:
            logfire.configure(token=token)
    except Exception:
        logging.exception("Tracing configuration failed")


configure_tracing()
base_url = setting("BACKEND_URL", "https://enterprise-agentic-rag-1-w3bs.onrender.com").strip().rstrip("/")
st.session_state.setdefault("session_id", str(uuid.uuid4()))
st.session_state.setdefault("messages", [])


def new_chat():
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.pop("pending_prompt", None)


def ask(prompt):
    st.session_state.pending_prompt = prompt


def show_answer(message):
    st.markdown(message["content"])
    sources = message.get("sources", [])
    if sources:
        with st.expander(f"Explore sources · {len(sources)} excerpts"):
            for index, source in enumerate(sources, 1):
                st.caption(f"EXCERPT {index}")
                st.markdown(str(source))
                if index < len(sources):
                    st.divider()


with st.sidebar:
    st.title("✦ Enterprise Assistant")
    st.caption("Your workspace for technical answers.")
    st.button("＋ New conversation", on_click=new_chat, width="stretch", type="primary")
    st.divider()
    st.markdown("**Explore your knowledge**")
    st.caption("Kubernetes · Intel hardware · Enterprise networking")
    st.markdown("**Make it a conversation**")
    st.caption("Ask a follow-up, request an example, or compare approaches. Your conversation stays in context.")
    if st.session_state.messages:
        transcript = "\n\n---\n\n".join(
            f"## {message['role'].title()}\n\n{message['content']}"
            for message in st.session_state.messages
        )
        st.download_button("Download conversation", transcript, file_name="enterprise-conversation.md", mime="text/markdown", width="stretch")

st.title("What would you like to explore?")
st.caption("Turn enterprise documentation into clear answers, practical examples, and next steps.")

if not st.session_state.messages:
    st.markdown("### Start with a question")
    examples = [
        ("☸ Kubernetes", "Understand your cluster", "Explain Kubernetes deployments, services, and pods with a practical example."),
        ("◈ Intel hardware", "Explore performance", "How does SR-IOV work with Intel network adapters?"),
        ("⇄ Networking", "Connect the concepts", "Compare VLANs and VXLANs and explain when to use each."),
    ]
    for column, (title, subtitle, question) in zip(st.columns(3), examples):
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(subtitle)
                st.button("Explore →", key=title, on_click=ask, args=(question,), width="stretch")
    st.info("Start with one of these topics, or write your own question below.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.warning(message["content"])
        elif message["role"] == "assistant":
            show_answer(message)
        else:
            st.markdown(message["content"])

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" and not st.session_state.messages[-1].get("error"):
    st.caption("KEEP EXPLORING")
    followups = [
        ("Explain simply", "Explain your last answer in simpler terms."),
        ("Show an example", "Give me a practical example based on your last answer."),
        ("Summarize", "Summarize your last answer in three key points."),
    ]
    for column, (label, question) in zip(st.columns(3), followups):
        column.button(label, on_click=ask, args=(question,), width="stretch")

entered_prompt = st.chat_input("Ask a question or follow up on an answer…")
prompt = st.session_state.pop("pending_prompt", None) or entered_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    error = None
    data = {}
    with st.chat_message("assistant"):
        with st.spinner("Working on your answer…"):
            try:
                response = requests.post(
                    f"{base_url}/query",
                    json={"q": prompt, "thread_id": st.session_state.session_id},
                    timeout=(10, 120),
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict) or not isinstance(data.get("answer"), str) or not data["answer"].strip() or data.get("status") == "error":
                    raise ValueError("Invalid answer received")
            except requests.exceptions.Timeout:
                error = "This answer took longer than expected. Please try again in a moment."
            except requests.exceptions.RequestException:
                logging.exception("Chat request failed")
                error = "The assistant is temporarily unavailable. Please try again in a moment."
            except (ValueError, TypeError):
                logging.exception("Invalid chat response")
                error = "We couldn't complete that answer. Please try rephrasing your question."
    if error:
        st.session_state.messages.append({"role": "assistant", "content": error, "error": True})
    else:
        sources = data.get("sources", [])
        st.session_state.messages.append({
            "role": "assistant",
            "content": data["answer"],
            "sources": sources if isinstance(sources, list) else [],
        })
    st.rerun()
