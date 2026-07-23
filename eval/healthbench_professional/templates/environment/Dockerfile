FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m pip install --no-cache-dir "openai>=2,<3"

WORKDIR /workspace

RUN mkdir -p /logs/agent /logs/verifier /workspace
