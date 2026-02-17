# ─────────────────────────────────────────────────────────────
#  Dockerfile — Agricultural Disease Diagnosis and Advisory System
#  Target: HuggingFace Spaces (Docker SDK)
#  Port: 7860 (Gradio default)
# ─────────────────────────────────────────────────────────────

FROM python:3.13-slim

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user (HF Spaces requirement) ────────────
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── Install Python deps (cached layer) ──────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project source ─────────────────────────────────────
COPY setup.py .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# ── Copy application + config files ─────────────────────────
COPY app.py .
COPY config/ config/
COPY params.yaml .
COPY knowledge/ knowledge/

# ── Copy model artifacts ─────────────────────────────────────
# NOTE: model.keras must be present at build time.
#       For HF Spaces, either:
#       1. Add model.keras to the HF Space repo (not gitignored there), or
#       2. Download from HF Hub at runtime (see scripts/download_model.py)
COPY artifacts/ artifacts/

# ── Feedback directory (writable) ────────────────────────────
RUN mkdir -p /app/outputs /app/logs && \
    chown -R appuser:appuser /app

# ── Switch to non-root user ─────────────────────────────────
USER appuser

# ── Expose Gradio port ───────────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# ── Launch ───────────────────────────────────────────────────
CMD ["python", "app.py"]
