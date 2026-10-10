# Streamlit backend 502 errors

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
