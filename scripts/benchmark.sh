#!/bin/bash
echo "📊 Ejecutando benchmark de SmartDoc Agent..."

# Verificar API
if ! curl -s http://localhost:8002/health > /dev/null; then
    echo "❌ API no está disponible en localhost:8002"
    echo "💡 Ejecuta primero: ./scripts/start-optimized.sh"
    exit 1
fi

echo "✅ API disponible - Iniciando benchmark..."

# Crear sesión de test
echo "🔧 Creando sesión de benchmark..."
SESSION_RESPONSE=$(curl -s -X POST http://localhost:8002/research/session \
    -H "Content-Type: application/json" \
    -d '{"topic":"benchmark test","objectives":["performance testing"],"optimization_level":"balanced"}')

SESSION_ID=$(echo $SESSION_RESPONSE | jq -r '.session_id')

if [ "$SESSION_ID" == "null" ]; then
    echo "❌ Error creando sesión de benchmark"
    echo "Respuesta: $SESSION_RESPONSE"
    exit 1
fi

echo "✅ Sesión creada: $SESSION_ID"

# Mostrar configuración
echo "🎯 Configuración de la sesión:"
echo $SESSION_RESPONSE | jq '{optimization_level, max_iterations, streaming_enabled, model}'

# Test queries con diferentes profundidades
declare -a test_queries=(
    "quick:¿Qué es Python?"
    "normal:Explica machine learning"
    "deep:Investiga las últimas tendencias en inteligencia artificial"
)

echo ""
echo "📋 Ejecutando queries de test..."
echo "================================"

for query_data in "${test_queries[@]}"; do
    IFS=':' read -r depth query <<< "$query_data"
    
    echo "🧪 Testing [$depth]: $query"
    
    start_time=$(date +%s.%3N)
    
    response=$(curl -s -X POST http://localhost:8002/research/chat/$SESSION_ID \
        -H "Content-Type: application/json" \
        -d "{\"message\":\"$query\",\"depth\":\"$depth\"}")
    
    end_time=$(date +%s.%3N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    success=$(echo $response | jq -r '.success // false')
    
    if [ "$success" == "true" ]; then
        echo "   ✅ Tiempo: ${duration}s"
        
        # Extraer métricas de performance
        if echo $response | jq -e '.performance' > /dev/null; then
            steps=$(echo $response | jq -r '.performance.total_steps // "N/A"')
            avg_step=$(echo $response | jq -r '.performance.avg_step_time // "N/A"')
            success_rate=$(echo $response | jq -r '.performance.success_rate // "N/A"')
            echo "   📊 Steps: $steps | Avg: ${avg_step}s | Success: $success_rate"
        fi
        
        # Mostrar primera línea de respuesta
        first_line=$(echo $response | jq -r '.response' | head -n1)
        echo "   💬 \"${first_line:0:60}...\""
    else
        echo "   ❌ FAILED en ${duration}s"
        error=$(echo $response | jq -r '.error // "Unknown error"')
        echo "   🚨 Error: $error"
    fi
    
    echo "   ---"
    sleep 3  # Pausa entre tests
done

# Información del sistema
echo ""
echo "🖥️ Información del hardware:"
curl -s http://localhost:8002/system/hardware | jq .

# Métricas finales de la sesión
echo ""
echo "📈 Métricas finales de la sesión:"
curl -s http://localhost:8002/research/session/$SESSION_ID/performance | jq .

echo ""
echo "📊 Benchmark completado exitosamente!"
