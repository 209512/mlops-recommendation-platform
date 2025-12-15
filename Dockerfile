# Multi-stage build for production optimization
FROM python:3.11-slim as builder

# Build arguments for version pinning
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install build dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Install uv for faster dependency management (pinned version for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.5.24 /uv /bin/uv

# Copy dependency files first for better cache utilization
COPY pyproject.toml uv.lock ./

# Install dependencies with uv using cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache -r uv.lock

# Production stage with minimal attack surface
FROM python:3.11-slim as production

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Security: Create appuser with specific UID/GID (non-root)
RUN groupadd -r appuser --gid=1001 && \
    useradd -r -g appuser --uid=1001 --home-dir /app --shell /bin/bash appuser

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code with proper ownership
COPY --chown=appuser:appuser app/ ./app/

# Switch to non-root user
USER appuser

# Security: Remove build tools and set proper permissions
RUN chmod -R 755 /app && \
    chmod -R 644 /app/app/*.py 2>/dev/null || true

# Expose port
EXPOSE 8000

# Enhanced health check with proper timeout and retry
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f -s --max-time 5 http://localhost:8000/health || exit 1

# Security: Signal handling for graceful shutdown
STOPSIGNAL SIGTERM

# Run with proper worker configuration
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]