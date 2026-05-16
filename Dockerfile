FROM python:3.12-slim

WORKDIR /app

# System deps for Pillow image transcoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app/src \
    PROXY_HOST=0.0.0.0 \
    PROXY_PORT=8788 \
    MODEL_CONFIG_PATH=/app/config/models.yaml

EXPOSE 8788
CMD ["sh", "-c", "PROXY_PORT=${PORT:-${PROXY_PORT:-8788}} python -m nvd_claude_proxy.main"]
