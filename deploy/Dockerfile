# CPU image for the TC-DANN API and Streamlit front ends.
# Weights are NOT baked in: mount a folder with the four *_model_best.joblib
# bundles at /app/models (see models/README.md).
#
#   docker build -f deploy/Dockerfile -t tc-dann .
#   docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" tc-dann
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-serve.txt

COPY tcdann/ tcdann/
COPY models/README.md models/README.md

ENV TC_DANN_BUNDLE_DIR=/app/models \
    SESSIONS_DIR=/app/sessions \
    PYTHONUNBUFFERED=1
WORKDIR /app/tcdann
EXPOSE 8000 8501
CMD ["uvicorn", "tc_dann_api_server:app", "--host", "0.0.0.0", "--port", "8000"]
