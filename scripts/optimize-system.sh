#!/bin/bash
echo "🚀 Optimizando sistema para SmartDoc Agent..."

# Verificar privilegios
if [ "$EUID" -ne 0 ]; then
    echo "❌ Este script necesita privilegios de root"
    echo "💡 Ejecuta: sudo ./scripts/optimize-system.sh"
    exit 1
fi

echo "⚙️ Aplicando optimizaciones del sistema..."

# Backup de archivos de configuración
echo "📦 Creando backup de configuraciones..."
cp /etc/sysctl.conf /etc/sysctl.conf.backup.$(date +%Y%m%d)
cp /etc/security/limits.conf /etc/security/limits.conf.backup.$(date +%Y%m%d)

# Optimizaciones de archivos abiertos
echo "📁 Optimizando límites de archivos..."
echo "fs.file-max = 65536" >> /etc/sysctl.conf
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimizaciones de red
echo "🌐 Optimizando red..."
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
echo "net.core.netdev_max_backlog = 5000" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65535" >> /etc/sysctl.conf

# Optimizaciones de memoria
echo "💾 Optimizando memoria..."
echo "vm.swappiness = 10" >> /etc/sysctl.conf
echo "vm.dirty_ratio = 15" >> /etc/sysctl.conf
echo "vm.dirty_background_ratio = 5" >> /etc/sysctl.conf

# Aplicar cambios
sysctl -p

# Optimizar Docker para GPU si está disponible
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 Optimizando Docker para GPU..."
    
    # Crear daemon.json optimizado
    cat > /etc/docker/daemon.json << 'DOCKER_EOF'
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "experimental": true
}
DOCKER_EOF
    
    echo "🔄 Reiniciando Docker..."
    systemctl restart docker
fi

echo ""
echo "✅ Optimizaciones del sistema aplicadas!"
echo "💡 Se recomienda reiniciar el sistema para efectos completos"
echo "🚀 Después del reinicio, ejecuta: ./scripts/start-optimized.sh"
