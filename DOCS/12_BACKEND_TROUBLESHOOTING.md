# Streamlit backend 502 errors

## Guardrails native crash fix

A Render log showing `Guardrails Check`, a Hugging Face model download, and
`Illegal instruction (core dumped)` identified a native crash during the first
guardrail request. Guardrails now explicitly uses NeMo's `google` embedding
provider with `gemini-embedding-001` instead of the default FastEmbed/ONNX model.
It uses the existing `GEMINI_API_KEY`; ensure this is set in Render. API errors
are raised rather than falling back to a local model or bypassing guardrails.
These embeddings are only for guardrail examples and do not change the Qdrant
collection or require re-ingesting documents.

Query stage markers now appear in Render's standard logs. Python fault handling
is enabled for native crashes, and Docker output is unbuffered. After deployment,
wait for `/health` to report `ready`, then test a greeting and a documentation
question. The first request will make remote embedding calls, so it can take
longer than subsequent requests.

## Checking backend status

The cloud UI calls a separate Render service. HTTP 502 means the gateway did
not receive a valid backend response; the status alone does not identify the
reason. A cold start, crashed process, or memory limit can produce this symptom.

After deploying the backend and UI changes, use **Check backend status** in the
Streamlit sidebar, or open `<BACKEND_URL>/health`:

| Response | Meaning | Next step |
| --- | --- | --- |
| 200, `ready` | Graph and guardrails initialized | Submit a query |
| 503, `starting` | Imports or model initialization still running | Wait, then check again |
| 503, `error` | Initialization raised a Python exception | Find `Backend initialization failed` and its traceback in Render logs |
| 502/504 or connection timeout | The application cannot respond through the gateway | Check Render deployment status, process logs, and memory usage |

The root route `/` is a liveness check. `/health` is a readiness check and can
be used as Render's health check path. Queries are rejected until initialization
completes; guardrails are not skipped to make the server appear ready.

Verify that Streamlit's `BACKEND_URL` points to the intended Render service.
The Docker image already binds Uvicorn to `0.0.0.0` and Render's `PORT`.
If using a native Python service, use the equivalent start command:

```sh
uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
```

Check required credentials against [environment variables](05_ENVIRONMENT_VARIABLES.md).
Do not paste credential values into logs or support messages.

If Render reports `Illegal instruction` or an out-of-memory termination, note
that disabling FlashRank does **not** remove all local inference: NeMo Guardrails
defaults to FastEmbed/ONNX for its example embeddings. Background initialization
cannot recover from a native process crash or operating-system memory kill.
Use the actual traceback or exit signal to decide whether the embedding runtime
or service resources need changing; increasing Streamlit's timeout will not
repair a crashed process.
