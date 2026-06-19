FROM ubuntu:24.04

# Install system dependencies including Docker CLI (needed by SWE-bench harness)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    docker.io \
    python3-dev \
    build-essential \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /mlops-assignment

# Copy dependency manifests first for better layer caching.
# The full project source is NOT copied here — it is either baked in below
# or mounted at runtime via docker-compose volumes.
COPY pyproject.toml .
COPY uv.lock .

# Install all Python dependencies (including Airflow, MLflow, mini-swe-agent).
# The .venv is created inside /mlops-assignment/.venv and is NOT hidden by any
# runtime volume mount (see docker-compose.yaml for the targeted mount strategy).
RUN uv sync --locked

# Copy scripts and dags so they are available inside the image.
COPY scripts scripts/
COPY dags dags/

RUN chmod +x scripts/*.sh

# Put the venv on PATH so 'airflow', 'uv', and 'python' all resolve correctly.
ENV PATH="/mlops-assignment/.venv/bin:$PATH"

# Airflow home — metadata DB, logs, and config live here inside the container.
ENV AIRFLOW_HOME="/opt/airflow"
