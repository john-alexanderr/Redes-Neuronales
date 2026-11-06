FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app

WORKDIR /workspace

COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src ./src

RUN python -m pip install --no-cache-dir -e ".[dev]"

COPY --chown=app:app . .

USER app

EXPOSE 8888
