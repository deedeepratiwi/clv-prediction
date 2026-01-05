FROM python:3.13.5-slim-bookworm

# Add uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (for Docker caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into /app/.venv
RUN uv sync --locked

# Ensure virtualenv binaries are used
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Copy entire project
COPY . .

# Expose API port
EXPOSE 8000

# Run FastAPI app
ENTRYPOINT ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
