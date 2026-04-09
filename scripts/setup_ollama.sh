#!/usr/bin/env bash
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Preparando modelo Llama en Ollama (Docker)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verificar que el contenedor esté corriendo
if ! docker ps | grep -q "polymarket_ollama"; then
    echo "Error: El contenedor 'polymarket_ollama' no está corriendo."
    echo "Ejecuta primero: docker compose up -d ollama"
    exit 1
fi

echo "Descargando modelo gemma3:4b (esto puede demorar unos minutos)..."
docker exec polymarket_ollama ollama pull gemma3:4b

echo "¡Modelo listo!"
echo "Ahora puedes iniciar los demás servicios:"
echo "  docker compose up -d"
