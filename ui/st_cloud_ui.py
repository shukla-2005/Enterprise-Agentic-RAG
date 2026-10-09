import os
import time
import uuid

import requests
import streamlit as st
import logfire


# =========================================================
# LOGFIRE CONFIGURATION
# =========================================================
try:
    logfire_token = st.secrets.get(
        "LOGFIRE_TOKEN",
        os.getenv("LOGFIRE_TOKEN")
    )

    if logfire_token:
        logfire.configure(token=logfire_token)
        logfire.instrument_requests()
        LOGFIRE_STATUS = "Connected & Tracing"
    else:
        LOGFIRE_STATUS = "Standby (No Token)"

except Exception as e:
    LOGFIRE_STATUS = "Standby (Configuration Failed)"
    print(f"Logfire configuration error: {e}")


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)


# =========================================================
# CONSTANTS
# =========================================================
AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# =========================================================
# BACKEND URL
# =========================================================
base_url = st.secrets.get(
    "BACKEND_URL",
    os.getenv(
        "BACKEND_URL",
        "https://enterprise-agentic-rag-1-w3bs.onrender.com"
    )
).rstrip("/")


# =========================================================
# SESSION MANAGEMENT
# =========================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

    try:
        logfire.info(
            "New User Session Created",
            session_id=st.session_state.session_id
        )
    except Exception:
        pass


if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("Agent OS")

    st.markdown("---")

    st.success(f"Logfire: {LOGFIRE_STATUS}")

    st.info(
        f"Memory ID: {st.session_state.session_id[:8]}"
    )

    st.caption(f"Backend: {base_url}")

    st.markdown("---")

    if st.button(
        "Clear History & Memory",
        width="stretch",
        type="primary"
    ):
        try:
            logfire.warning(
                "Memory Wipe Triggered",
                session_id=st.session_state.session_id
            )
        except Exception:
            pass

        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())

        st.rerun()


# =========================================================
# MAIN CHAT
# =========================================================
st.title("Enterprise Agentic Assistant")

st.caption(
    "Ask questions about your enterprise documentation."
)


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================
for message in st.session_state.messages:

    avatar = (
        AI_AVATAR
        if message["role"] == "assistant"
        else USER_AVATAR
    )

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):
        st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================
prompt = st.chat_input(
    "Ask about your documentation..."
)


if prompt:

    # -----------------------------------------------------
    # Save user message
    # -----------------------------------------------------
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message(
        "user",
        avatar=USER_AVATAR
    ):
        st.markdown(prompt)


    # -----------------------------------------------------
    # USER TRACE
    # -----------------------------------------------------
    try:
        trace_context = logfire.span(
            "User Chat Interaction",
            user_query=prompt,
            session_id=st.session_state.session_id
        )
    except Exception:
        trace_context = None


    # -----------------------------------------------------
    # ASSISTANT RESPONSE
    # -----------------------------------------------------
    with st.chat_message(
        "assistant",
        avatar=AI_AVATAR
    ):

        data = {}

        request_failed = False
        error_message = ""

        # =================================================
        # BACKEND REQUEST STATUS
        # =================================================
        with st.status(
            "Agent is thinking...",
            expanded=True
        ) as status:

            try:

                url = f"{base_url}/query"

                payload = {
                    "q": prompt,
                    "thread_id": st.session_state.session_id
                }

                st.write("Connecting to RAG backend...")

                try:
                    with logfire.span(
                        "Calling RAG Backend",
                        backend_url=url
                    ):

                        response = requests.post(
                            url,
                            json=payload,
                            timeout=120
                        )

                except Exception:
                    response = requests.post(
                        url,
                        json=payload,
                        timeout=120
                    )

                # -----------------------------------------
                # Non-200 backend response
                # -----------------------------------------
                if response.status_code != 200:

                    request_failed = True

                    try:
                        backend_body = response.text
                    except Exception:
                        backend_body = "No backend response body."

                    error_message = (
                        f"Backend Error: "
                        f"{response.status_code}\n\n"
                        f"{backend_body}"
                    )

                    status.update(
                        label="Backend Error",
                        state="error",
                        expanded=True
                    )

                else:

                    # -------------------------------------
                    # Parse backend JSON
                    # -------------------------------------
                    try:
                        data = response.json()

                    except ValueError:

                        request_failed = True

                        error_message = (
                            "Backend returned an invalid "
                            "JSON response.\n\n"
                            f"Response:\n{response.text}"
                        )

                        status.update(
                            label="Invalid Backend Response",
                            state="error",
                            expanded=True
                        )


                    # -------------------------------------
                    # Thought Process
                    # -------------------------------------
                    if not request_failed:

                        steps = data.get(
                            "thought_process",
                            []
                        )

                        if steps:

                            st.markdown(
                                "#### Agent Progress"
                            )

                            for step in steps:

                                st.markdown(
                                    str(step),
                                    unsafe_allow_html=False
                                )

                        status.update(
                            label="Answer Synthesized",
                            state="complete",
                            expanded=False
                        )


            # =================================================
            # TIMEOUT
            # =================================================
            except requests.exceptions.Timeout:

                request_failed = True

                error_message = (
                    "Backend request timed out.\n\n"
                    "The Render backend may be waking up "
                    "or the RAG pipeline is taking too long."
                )

                status.update(
                    label="Backend Timeout",
                    state="error",
                    expanded=True
                )


            # =================================================
            # CONNECTION ERROR
            # =================================================
            except requests.exceptions.ConnectionError as e:

                request_failed = True

                error_message = (
                    "Unable to connect to the backend.\n\n"
                    f"{e}"
                )

                try:
                    logfire.error(
                        "Backend Connection Error",
                        error=str(e),
                        backend_url=base_url
                    )
                except Exception:
                    pass

                status.update(
                    label="Connection Failed",
                    state="error",
                    expanded=True
                )


            # =================================================
            # GENERAL REQUEST ERROR
            # =================================================
            except requests.exceptions.RequestException as e:

                request_failed = True

                error_message = (
                    "Backend request failed.\n\n"
                    f"{e}"
                )

                try:
                    logfire.error(
                        "UI Backend Request Failed",
                        error=str(e)
                    )
                except Exception:
                    pass

                status.update(
                    label="Request Failed",
                    state="error",
                    expanded=True
                )


            # =================================================
            # UNEXPECTED ERROR
            # =================================================
            except Exception as e:

                request_failed = True

                error_message = (
                    "Unexpected frontend error.\n\n"
                    f"{type(e).__name__}: {e}"
                )

                try:
                    logfire.error(
                        "Unexpected UI Error",
                        error=str(e)
                    )
                except Exception:
                    pass

                status.update(
                    label="Unexpected Error",
                    state="error",
                    expanded=True
                )


        # =====================================================
        # IMPORTANT:
        # st.stop() is OUTSIDE st.status()
        # =====================================================
        if request_failed:

            st.error(error_message)

            try:
                logfire.error(
                    "Chat Request Failed",
                    error=error_message,
                    session_id=st.session_state.session_id
                )
            except Exception:
                pass

            st.stop()


        # =====================================================
        # ANSWER
        # =====================================================
        full_answer = data.get(
            "answer",
            "No response received from the backend."
        )

        answer_placeholder = st.empty()

        current_text = ""

        for char in full_answer:

            current_text += char

            answer_placeholder.markdown(
                current_text + "▌"
            )

            time.sleep(0.005)

        answer_placeholder.markdown(
            full_answer
        )


        # =====================================================
        # SOURCES
        # =====================================================
        sources = data.get(
            "sources",
            []
        )

        if sources:

            with st.expander(
                f"Retrieved Context "
                f"({len(sources)} chunks)"
            ):

                for index, source in enumerate(
                    sources,
                    start=1
                ):

                    st.caption(
                        f"Chunk {index}"
                    )

                    st.info(
                        str(source)
                    )

        else:

            st.caption(
                "No context retrieved — "
                "conversational response."
            )


        # =====================================================
        # SAVE ASSISTANT MESSAGE
        # =====================================================
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_answer
            }
        )

        try:
            logfire.info(
                "Chat cycle completed successfully",
                session_id=st.session_state.session_id
            )
        except Exception:
            pass