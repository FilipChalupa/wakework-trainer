# ---------- 1) frontend build ----------
FROM node:22-alpine AS frontend
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------- 2) backend + trainer ----------
FROM python:3.11-slim AS backend

# CPU by default; docker-compose.gpu.yml overrides this with "tensorflow[and-cuda]==2.19.0"
ARG TENSORFLOW_PACKAGE="tensorflow==2.19.0"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --upgrade pip \
    && pip install "${TENSORFLOW_PACKAGE}" \
    && pip install -r requirements.txt

COPY backend/ ./
COPY --from=frontend /app/dist ./static

VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
