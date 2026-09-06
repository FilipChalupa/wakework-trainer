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
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git libsndfile1 gosu \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 app && useradd -m -u 1000 -g app app

WORKDIR /app
COPY backend/requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install "${TENSORFLOW_PACKAGE}" \
    && pip install -r requirements.txt

# microWakeWord itself: editable install from a pinned git checkout (its sub-packages have no __init__.py,
# so a regular wheel build would silently drop microwakeword.audio / microwakeword.layers)
ARG MICROWAKEWORD_REF=4665173cd35f1cff9a61e06fc427f124766c488e
RUN git clone https://github.com/kahrendt/microWakeWord /opt/microWakeWord \
    && git -C /opt/microWakeWord checkout --quiet "${MICROWAKEWORD_REF}" \
    && touch /opt/microWakeWord/microwakeword/audio/__init__.py /opt/microWakeWord/microwakeword/layers/__init__.py \
    && pip install --no-deps -e /opt/microWakeWord \
    && python -c "import microwakeword.audio.augmentation, microwakeword.layers.modes, microwakeword.model_train_eval"

COPY backend/ ./
COPY --from=frontend /app/dist ./static
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && chown -R app:app /app

ENV HOME=/home/app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
