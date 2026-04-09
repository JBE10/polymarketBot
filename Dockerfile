FROM python:3.12-slim

# Evitar que pip escriba bytecode y forzar output unbuffered
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

WORKDIR /app

# Dependencias de sistema necesarias para compilar paquetes C y curl para healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Puertos: 8080 (bot healthcheck), 8501 (dashboard)
EXPOSE 8080 8501

CMD ["python", "-m", "src.main"]
