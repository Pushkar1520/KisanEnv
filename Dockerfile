
FROM python:3.11-slim-bookworm AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml requirements.txt ./

RUN uv pip install --system torch

RUN uv pip install --system \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    openenv \
    numpy==1.26.4 \
    transformers>=4.46.0 \
    trl \
    datasets==2.19.0 \
    accelerate \
    peft \
    scipy==1.13.0 \
    httpx==0.27.0 \
    rich==13.7.0 \
    matplotlib==3.9.0 \
    bitsandbytes

COPY . .

RUN mkdir -p checkpoints agents ui

ENV KISANENV_LLM_BACKEND=huggingface
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "7860"]
