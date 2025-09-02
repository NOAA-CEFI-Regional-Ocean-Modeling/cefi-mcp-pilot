# Use the official image with uv included
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for netcdf4 and other scientific packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libnetcdf-dev \
    libhdf5-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and source code
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install dependencies and the project
RUN uv sync --frozen --no-dev

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose default MCP HTTP port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default command to run the MCP server with HTTP transport
CMD ["uv", "run", "python", "-m", "cefi_mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
