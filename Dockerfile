FROM python:3.12-slim

# Unbuffered, so `docker logs` shows what happened before a crash; no .pyc
# files, because the image is read-only in spirit and they are noise in `ls`.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

LABEL org.opencontainers.image.title="GPU control panel" \
      org.opencontainers.image.description="Arbitrates a GPU between llama.cpp and the consumers that want it." \
      org.opencontainers.image.source="https://github.com/Santiago-G1/gpupanel" \
      org.opencontainers.image.documentation="https://github.com/Santiago-G1/gpupanel#readme"

# openssh-client is the transport to the hosts this panel drives; the image
# runs as root only so that the mounted key is reachable at /root/.ssh.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY static ./static

EXPOSE 8099
# /api/health answers from the last card probe, without an SSH of its own, so
# this says whether the panel can still see the GPU host. The container's own
# address, not 127.0.0.1: it also fails when uvicorn bound only loopback.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import socket, urllib.request; h = socket.gethostbyname(socket.gethostname()); urllib.request.urlopen(f'http://{h}:8099/api/health', timeout=5)"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8099"]
