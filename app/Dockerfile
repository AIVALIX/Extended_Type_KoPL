FROM python:3.10-slim

WORKDIR /app

# ---- システムパッケージが必要ならここに apt-get ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# ---- Python ----
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- アプリ本体 ----
COPY . .

EXPOSE 8001
