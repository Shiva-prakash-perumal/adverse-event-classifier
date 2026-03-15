# ─── Base image ───────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ─── Metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="Shiva Prakash Perumal"
LABEL description="Adverse Event Intelligence Pipeline"
LABEL version="1.0"

# ─── Environment ──────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ─── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Python dependencies ──────────────────────────────────────────────────────
# Copy requirements first for Docker layer caching
# (Only re-installs if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Copy application code ────────────────────────────────────────────────────
COPY . .

# ─── Create necessary directories ─────────────────────────────────────────────
RUN mkdir -p data models mlruns

# ─── Expose Streamlit port ────────────────────────────────────────────────────
EXPOSE 8501

# ─── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ─── Default command: run Streamlit app ───────────────────────────────────────
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
