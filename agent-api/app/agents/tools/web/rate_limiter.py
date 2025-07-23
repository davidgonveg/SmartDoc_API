"""
Rate Limiter for Web Search Tool
Sistema de rate limiting para respetar límites de APIs y sitios web
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import redis.asyncio as redis
import json

logger = logging.getLogger(__name__)

class RateLimitType(Enum):
    """Tipos de rate limiting"""
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    CONCURRENT_REQUESTS = "concurrent_requests"
    DELAY_BETWEEN_REQUESTS = "delay_between_requests"

@dataclass
class RateLimit:
    """Configuración de rate limit"""
    limit_type: RateLimitType
    limit: int  # Número máximo de requests
    window: int  # Ventana de tiempo en segundos
    burst_limit: Optional[int] = None  # Límite de burst (opcional)
    backoff_factor: float = 1.5  # Factor de backoff exponencial
    max_delay: int = 300  # Delay máximo en segundos (5 minutos)

@dataclass
class RateLimitState:
    """Estado actual de rate limiting"""
    requests_made: int = 0
    window_start: float = field(default_factory=time.time)
    last_request_time: float = 0
    consecutive_violations: int = 0
    current_delay: float = 0
    blocked_until: Optional[float] = None
    request_timestamps: List[float] = field(default_factory=list)

class InMemoryRateLimiter:
    """Rate limiter en memoria (para desarrollo/testing)"""
    
    def __init__(self):
        self.states: Dict[str, RateLimitState] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
    
    async def get_state(self, key: str) -> RateLimitState:
        """Obtener estado de rate limiting para una clave"""
        if key not in self.states:
            self.states[key] = RateLimitState()
        return self.states[key]
    
    async def update_state(self, key: str, state: RateLimitState):
        """Actualizar estado de rate limiting"""
        self.states[key] = state
    
    async def get_lock(self, key: str) -> asyncio.Lock:
        """Obtener lock para una clave"""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        return self.locks[key]
    
    async def cleanup_old_data(self):
        """Limpiar datos antiguos (llamar periódicamente)"""
        current_time = time.time()
        cutoff_time = current_time - 86400  # 24 horas
        
        keys_to_remove = []
        for key, state in self.states.items():
            if state.last_request_time < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.states[key]
            if key in self.locks:
                del self.locks[key]

class RedisRateLimiter:
    """Rate limiter usando Redis (para producción)"""
    
    def __init__(self, redis_client: redis.Redis, key_prefix: str = "rate_limit:"):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.locks: Dict[str, asyncio.Lock] = {}
    
    async def get_state(self, key: str) -> RateLimitState:
        """Obtener estado desde Redis"""
        redis_key = f"{self.key_prefix}{key}"
        
        try:
            data = await self.redis.get(redis_key)
            if data:
                state_dict = json.loads(data)
                return RateLimitState(
                    requests_made=state_dict.get('requests_made', 0),
                    window_start=state_dict.get('window_start', time.time()),
                    last_request_time=state_dict.get('last_request_time', 0),
                    consecutive_violations=state_dict.get('consecutive_violations', 0),
                    current_delay=state_dict.get('current_delay', 0),
                    blocked_until=state_dict.get('blocked_until'),
                    request_timestamps=state_dict.get('request_timestamps', [])
                )
            else:
                return RateLimitState()
        
        except Exception as e:
            logger.warning(f"Error reading rate limit state from Redis: {e}")
            return RateLimitState()
    
    async def update_state(self, key: str, state: RateLimitState):
        """Actualizar estado en Redis"""
        redis_key = f"{self.key_prefix}{key}"
        
        try:
            state_dict = {
                'requests_made': state.requests_made,
                'window_start': state.window_start,
                'last_request_time': state.last_request_time,
                'consecutive_violations': state.consecutive_violations,
                'current_delay': state.current_delay,
                'blocked_until': state.blocked_until,
                'request_timestamps': state.request_timestamps
            }
            
            # Guardar con TTL de 24 horas
            await self.redis.setex(
                redis_key, 
                86400, 
                json.dumps(state_dict)
            )
        
        except Exception as e:
            logger.error(f"Error updating rate limit state in Redis: {e}")
    
    async def get_lock(self, key: str) -> asyncio.Lock:
        """Obtener lock local (Redis locks son más complejos)"""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        return self.locks[key]

class RateLimitManager:
    """Gestor principal de rate limiting"""
    
    def __init__(self, 
                 backend: Optional[Any] = None,
                 global_rate_limit: int = 100,
                 default_delay: float = 1.0):
        
        self.backend = backend or InMemoryRateLimiter()
        self.global_rate_limit = global_rate_limit
        self.default_delay = default_delay
        
        # Configuraciones por dominio/engine
        self.domain_limits: Dict[str, List[RateLimit]] = {}
        self.engine_limits: Dict[str, List[RateLimit]] = {}
        
        # Estadísticas globales
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'total_delay_time': 0,
            'violations_by_domain': {},
        }
        
        self._setup_default_limits()
    
    def _setup_default_limits(self):
        """Configurar límites por defecto"""
        
        # Límites globales
        self.global_limits = [
            RateLimit(RateLimitType.REQUESTS_PER_MINUTE, self.global_rate_limit, 60),
            RateLimit(RateLimitType.CONCURRENT_REQUESTS, 5, 0),
            RateLimit(RateLimitType.DELAY_BETWEEN_REQUESTS, 1, 0)
        ]
        
        # Límites específicos por dominio popular
        self.domain_limits.update({
            'google.com': [
                RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 10, 60),
                RateLimit(RateLimitType.DELAY_BETWEEN_REQUESTS, 2, 0)
            ],
            'duckduckgo.com': [
                RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 30, 60),
                RateLimit(RateLimitType.DELAY_BETWEEN_REQUESTS, 1, 0)
            ],
            'wikipedia.org': [
                RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 60, 60),
                RateLimit(RateLimitType.DELAY_BETWEEN_REQUESTS, 0.5, 0)
            ],
            'github.com': [
                RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 60, 60),
                RateLimit(RateLimitType.REQUESTS_PER_HOUR, 1000, 3600)
            ],
            'stackoverflow.com': [
                RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 30, 60),
                RateLimit(RateLimitType.DELAY_BETWEEN_REQUESTS, 1, 0)
            ],
        })
        
        # Límites por motor de búsqueda
        self.engine_limits.update({
            'duckduckgo': [
                RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 30, 60),
                RateLimit(RateLimitType.DELAY_BETWEEN_REQUESTS, 1, 0)
            ],
            'google_custom': [
                RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 100, 60),
                RateLimit(RateLimitType.REQUESTS_PER_DAY, 1000, 86400)
            ],
            'bing': [
                RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 1000, 60),
                RateLimit(RateLimitType.REQUESTS_PER_DAY, 10000, 86400)
            ],
            'searx': [
                RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 60, 60),
                RateLimit(RateLimitType.DELAY_BETWEEN_REQUESTS, 1, 0)
            ],
        })
    
    def add_domain_limit(self, domain: str, rate_limit: RateLimit):
        """Agregar límite personalizado para un dominio"""
        if domain not in self.domain_limits:
            self.domain_limits[domain] = []
        self.domain_limits[domain].append(rate_limit)
    
    def add_engine_limit(self, engine: str, rate_limit: RateLimit):
        """Agregar límite personalizado para un motor de búsqueda"""
        if engine not in self.engine_limits:
            self.engine_limits[engine] = []
        self.engine_limits[engine].append(rate_limit)
    
    async def acquire(self, 
                     identifier: str,
                     domain: Optional[str] = None,
                     engine: Optional[str] = None) -> bool:
        """
        Intentar adquirir permiso para hacer request
        Retorna True si se puede proceder, False si está bloqueado
        """
        
        current_time = time.time()
        
        # Obtener límites aplicables
        applicable_limits = self.global_limits.copy()
        
        if domain and domain in self.domain_limits:
            applicable_limits.extend(self.domain_limits[domain])
        
        if engine and engine in self.engine_limits:
            applicable_limits.extend(self.engine_limits[engine])
        
        # Verificar cada límite
        for rate_limit in applicable_limits:
            key = f"{identifier}:{rate_limit.limit_type.value}"
            
            async with await self.backend.get_lock(key):
                state = await self.backend.get_state(key)
                
                # Verificar si está bloqueado
                if state.blocked_until and current_time < state.blocked_until:
                    logger.debug(f"Request blocked until {state.blocked_until}")
                    return False
                
                # Verificar límite específico
                if not await self._check_rate_limit(rate_limit, state, current_time):
                    # Calcular tiempo de bloqueo
                    delay = await self._calculate_delay(rate_limit, state)
                    state.blocked_until = current_time + delay
                    state.consecutive_violations += 1
                    
                    await self.backend.update_state(key, state)
                    
                    # Actualizar estadísticas
                    self.stats['blocked_requests'] += 1
                    if domain:
                        if domain not in self.stats['violations_by_domain']:
                            self.stats['violations_by_domain'][domain] = 0
                        self.stats['violations_by_domain'][domain] += 1
                    
                    logger.warning(f"Rate limit exceeded for {identifier}, blocked for {delay}s")
                    return False
        
        # Si pasó todas las verificaciones, actualizar estados
        for rate_limit in applicable_limits:
            key = f"{identifier}:{rate_limit.limit_type.value}"
            
            async with await self.backend.get_lock(key):
                state = await self.backend.get_state(key)
                await self._update_rate_limit_state(rate_limit, state, current_time)
                await self.backend.update_state(key, state)
        
        # Actualizar estadísticas globales
        self.stats['total_requests'] += 1
        
        return True
    
    async def _check_rate_limit(self, 
                               rate_limit: RateLimit, 
                               state: RateLimitState, 
                               current_time: float) -> bool:
        """Verificar si un rate limit específico se cumple"""
        
        if rate_limit.limit_type == RateLimitType.REQUESTS_PER_MINUTE:
            return await self._check_windowed_limit(rate_limit, state, current_time, 60)
        
        elif rate_limit.limit_type == RateLimitType.REQUESTS_PER_HOUR:
            return await self._check_windowed_limit(rate_limit, state, current_time, 3600)
        
        elif rate_limit.limit_type == RateLimitType.REQUESTS_PER_DAY:
            return await self._check_windowed_limit(rate_limit, state, current_time, 86400)
        
        elif rate_limit.limit_type == RateLimitType.DELAY_BETWEEN_REQUESTS:
            if state.last_request_time == 0:
                return True
            time_since_last = current_time - state.last_request_time
            return time_since_last >= rate_limit.limit
        
        elif rate_limit.limit_type == RateLimitType.CONCURRENT_REQUESTS:
            # Este requiere lógica más compleja, por ahora siempre permitir
            return True
        
        return True
    
    async def _check_windowed_limit(self, 
                                   rate_limit: RateLimit, 
                                   state: RateLimitState, 
                                   current_time: float,
                                   window_seconds: int) -> bool:
        """Verificar límite basado en ventana de tiempo"""
        
        # Limpiar timestamps antiguos
        cutoff_time = current_time - window_seconds
        state.request_timestamps = [
            ts for ts in state.request_timestamps 
            if ts > cutoff_time
        ]
        
        # Verificar si se puede hacer otro request
        return len(state.request_timestamps) < rate_limit.limit
    
    async def _update_rate_limit_state(self, 
                                      rate_limit: RateLimit, 
                                      state: RateLimitState, 
                                      current_time: float):
        """Actualizar state después de un request exitoso"""
        
        state.last_request_time = current_time
        state.requests_made += 1
        state.consecutive_violations = 0  # Reset violations
        state.blocked_until = None  # Clear any block
        
        # Agregar timestamp para límites basados en ventana
        if rate_limit.limit_type in [
            RateLimitType.REQUESTS_PER_MINUTE,
            RateLimitType.REQUESTS_PER_HOUR,
            RateLimitType.REQUESTS_PER_DAY
        ]:
            state.request_timestamps.append(current_time)
            
            # Mantener solo timestamps recientes (optimización)
            max_window = 86400  # 24 horas
            cutoff_time = current_time - max_window
            state.request_timestamps = [
                ts for ts in state.request_timestamps 
                if ts > cutoff_time
            ]
    
    async def _calculate_delay(self, rate_limit: RateLimit, state: RateLimitState) -> float:
        """Calcular delay de bloqueo"""
        
        base_delay = rate_limit.window if rate_limit.window > 0 else self.default_delay
        
        # Exponential backoff basado en violaciones consecutivas
        delay = base_delay * (rate_limit.backoff_factor ** state.consecutive_violations)
        
        # Aplicar límite máximo
        return min(delay, rate_limit.max_delay)
    
    async def wait_if_needed(self, 
                            identifier: str,
                            domain: Optional[str] = None,
                            engine: Optional[str] = None,
                            max_wait: int = 60) -> bool:
        """
        Esperar si es necesario hasta poder hacer request
        Retorna True si se puede proceder, False si timeout
        """
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if await self.acquire(identifier, domain, engine):
                return True
            
            # Esperar un poco antes de reintentar
            await asyncio.sleep(1)
        
        logger.warning(f"Timeout waiting for rate limit clearance: {identifier}")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rate limiting"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Resetear estadísticas"""
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'total_delay_time': 0,
            'violations_by_domain': {},
        }
    
    async def get_current_limits(self, 
                                identifier: str,
                                domain: Optional[str] = None,
                                engine: Optional[str] = None) -> Dict[str, Any]:
        """Obtener límites actuales y estado para un identificador"""
        
        applicable_limits = self.global_limits.copy()
        
        if domain and domain in self.domain_limits:
            applicable_limits.extend(self.domain_limits[domain])
        
        if engine and engine in self.engine_limits:
            applicable_limits.extend(self.engine_limits[engine])
        
        limits_info = {}
        current_time = time.time()
        
        for rate_limit in applicable_limits:
            key = f"{identifier}:{rate_limit.limit_type.value}"
            state = await self.backend.get_state(key)
            
            limits_info[rate_limit.limit_type.value] = {
                'limit': rate_limit.limit,
                'window': rate_limit.window,
                'requests_made': len(state.request_timestamps) if hasattr(state, 'request_timestamps') else state.requests_made,
                'remaining': max(0, rate_limit.limit - len(state.request_timestamps)) if hasattr(state, 'request_timestamps') else None,
                'reset_time': state.window_start + rate_limit.window if rate_limit.window > 0 else None,
                'blocked_until': state.blocked_until,
                'is_blocked': state.blocked_until and current_time < state.blocked_until,
            }
        
        return limits_info

# Factory para crear rate limiter
def create_rate_limiter(redis_url: Optional[str] = None, **kwargs) -> RateLimitManager:
    """Factory para crear rate limiter con backend apropiado"""
    
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url)
            backend = RedisRateLimiter(redis_client)
            logger.info("Using Redis backend for rate limiting")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis, using in-memory backend: {e}")
            backend = InMemoryRateLimiter()
    else:
        backend = InMemoryRateLimiter()
        logger.info("Using in-memory backend for rate limiting")
    
    return RateLimitManager(backend=backend, **kwargs)

# Instancia global
global_rate_limiter = create_rate_limiter()