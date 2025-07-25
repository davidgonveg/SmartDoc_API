#!/bin/bash
echo "🛑 Parando SmartDoc Agent..."

echo "🐳 Parando servicios Docker..."
docker-compose down

echo "🧹 Limpiando recursos..."
docker system prune -f --volumes

echo "✅ SmartDoc Agent parado correctamente"
echo "🚀 Para reiniciar: ./scripts/start-optimized.sh"
