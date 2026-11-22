FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BIOAGENT_CACHE_PATH=/app/.cache/bioagent.sqlite

WORKDIR /app

# Dependencies first so code edits don't invalidate the install layer.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[app]"

COPY app ./app
COPY evals ./evals

RUN useradd --create-home --uid 10001 bioagent \
    && mkdir -p /app/.cache \
    && chown -R bioagent /app/.cache
USER bioagent

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

# Secrets come from the environment at run time: docker run --env-file .env ...
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
