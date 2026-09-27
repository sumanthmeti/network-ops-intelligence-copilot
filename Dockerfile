# API image: FastAPI + LangGraph agent + the trained anomaly model.
#
# Build (after `python ml/experiment.py` has exported artifacts/model/):
#   docker build -t noc-copilot-api:latest .
#   docker run -p 8000:8000 noc-copilot-api:latest

# ---- stage 1: install dependencies into a virtualenv -------------------------------
# Separate stage so compilers / pip caches never reach the final image.
FROM python:3.12-slim AS deps
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Copy ONLY the requirements first: this layer is cached until requirements change,
# so editing application code doesn't re-download every package.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# ---- stage 2: runtime --------------------------------------------------------------
FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/model/model.joblib \
    CHANGE_CALENDAR_PATH=/app/data/raw/change_calendar.json \
    LLM_BACKEND=offline \
    DETECTOR_BACKEND=local

# Non-root user: if the app is compromised, the attacker isn't root in the container.
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=deps /opt/venv /opt/venv

# Only what the service needs at runtime.
COPY agent/ agent/
COPY api/ api/
COPY ml/features.py ml/train.py ml/
COPY artifacts/model/ artifacts/model/
COPY data/raw/change_calendar.json data/raw/change_calendar.json

USER app
EXPOSE 8000

# Docker-level health check (Kubernetes uses its own probes from the manifest instead).
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

# Exec form: uvicorn is PID 1 and receives SIGTERM directly, so Kubernetes rolling
# updates shut it down gracefully instead of killing it after the grace period.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
