"""
User Agent Management for Web Search Tool
Gestión de User-Agents para evitar bloqueos y detectar bots
"""

import random
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class BrowserType(Enum):
    """Tipos de navegadores disponibles"""
    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"
    OPERA = "opera"

class PlatformType(Enum):
    """Tipos de plataformas/OS"""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    ANDROID = "android"
    IOS = "ios"

@dataclass
class UserAgentInfo:
    """Información de un User-Agent"""
    user_agent: str
    browser: BrowserType
    platform: PlatformType
    version: str
    popularity_score: float  # 0-1, mayor = más común
    last_used: Optional[float] = None
    use_count: int = 0

# Pool completo de User-Agents reales y actuales
USER_AGENT_POOL = [
    # Chrome Windows (más populares)
    UserAgentInfo(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        BrowserType.CHROME, PlatformType.WINDOWS, "120.0", 0.95
    ),
    UserAgentInfo(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        BrowserType.CHROME, PlatformType.WINDOWS, "119.0", 0.90
    ),
    UserAgentInfo(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        BrowserType.CHROME, PlatformType.WINDOWS, "118.0", 0.85
    ),
    
    # Chrome macOS
    UserAgentInfo(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        BrowserType.CHROME, PlatformType.MACOS, "120.0", 0.80
    ),
    UserAgentInfo(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        BrowserType.CHROME, PlatformType.MACOS, "119.0", 0.75
    ),
    
    # Chrome Linux
    UserAgentInfo(
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        BrowserType.CHROME, PlatformType.LINUX, "120.0", 0.70
    ),
    UserAgentInfo(
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        BrowserType.CHROME, PlatformType.LINUX, "119.0", 0.65
    ),
    
    # Firefox Windows
    UserAgentInfo(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        BrowserType.FIREFOX, PlatformType.WINDOWS, "121.0", 0.80
    ),
    UserAgentInfo(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        BrowserType.FIREFOX, PlatformType.WINDOWS, "120.0", 0.75
    ),
    UserAgentInfo(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0",
        BrowserType.FIREFOX, PlatformType.WINDOWS, "119.0", 0.70
    ),
    
    # Firefox macOS
    UserAgentInfo(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        BrowserType.FIREFOX, PlatformType.MACOS, "121.0", 0.65
    ),
    UserAgentInfo(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
        BrowserType.FIREFOX, PlatformType.MACOS, "120.0", 0.60
    ),
    
    # Firefox Linux
    UserAgentInfo(
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
        BrowserType.FIREFOX, PlatformType.LINUX, "121.0", 0.55
    ),
    UserAgentInfo(
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
        BrowserType.FIREFOX, PlatformType.LINUX, "121.0", 0.50
    ),
    
    # Safari macOS
    UserAgentInfo(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        BrowserType.SAFARI, PlatformType.MACOS, "17.1", 0.70
    ),
    UserAgentInfo(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        BrowserType.SAFARI, PlatformType.MACOS, "17.0", 0.65
    ),
    UserAgentInfo(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
        BrowserType.SAFARI, PlatformType.MACOS, "16.6", 0.60
    ),
    
    # Edge Windows
    UserAgentInfo(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        BrowserType.EDGE, PlatformType.WINDOWS, "120.0", 0.75
    ),
    UserAgentInfo(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
        BrowserType.EDGE, PlatformType.WINDOWS, "119.0", 0.70
    ),
    
    # Edge macOS
    UserAgentInfo(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        BrowserType.EDGE, PlatformType.MACOS, "120.0", 0.60
    ),
    
    # Mobile User Agents (para diversidad)
    UserAgentInfo(
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
        BrowserType.SAFARI, PlatformType.IOS, "17.1", 0.55
    ),
    UserAgentInfo(
        "Mozilla/5.0 (Linux; Android 14; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        BrowserType.CHROME, PlatformType.ANDROID, "120.0", 0.50
    ),
    
    # Opera (menos común pero útil para diversidad)
    UserAgentInfo(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
        BrowserType.OPERA, PlatformType.WINDOWS, "106.0", 0.40
    ),
    UserAgentInfo(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
        BrowserType.OPERA, PlatformType.MACOS, "106.0", 0.35
    ),
]

# User-Agents específicos para búsquedas académicas/profesionales
ACADEMIC_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# User-Agents para sitios específicos que son más restrictivos
SITE_SPECIFIC_USER_AGENTS = {
    "google.com": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ],
    "linkedin.com": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ],
    "twitter.com": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    ],
}

class UserAgentManager:
    """Gestor de User-Agents con rotación inteligente"""
    
    def __init__(self, pool: List[UserAgentInfo] = None):
        self.pool = pool or USER_AGENT_POOL.copy()
        self.current_index = 0
        self.last_rotation_time = time.time()
        self.rotation_interval = 300  # 5 minutos
        self.usage_stats = {}
        
    def get_random_user_agent(self, 
                            browser_type: Optional[BrowserType] = None,
                            platform_type: Optional[PlatformType] = None,
                            site_domain: Optional[str] = None) -> str:
        """Obtener User-Agent aleatorio con filtros opcionales"""
        
        # Si hay User-Agents específicos para el sitio
        if site_domain and site_domain in SITE_SPECIFIC_USER_AGENTS:
            return random.choice(SITE_SPECIFIC_USER_AGENTS[site_domain])
        
        # Filtrar pool según criterios
        filtered_pool = self.pool
        
        if browser_type:
            filtered_pool = [ua for ua in filtered_pool if ua.browser == browser_type]
        
        if platform_type:
            filtered_pool = [ua for ua in filtered_pool if ua.platform == platform_type]
        
        if not filtered_pool:
            filtered_pool = self.pool  # Fallback al pool completo
        
        # Selección ponderada por popularidad
        weights = [ua.popularity_score for ua in filtered_pool]
        selected_ua = random.choices(filtered_pool, weights=weights, k=1)[0]
        
        # Actualizar estadísticas
        selected_ua.last_used = time.time()
        selected_ua.use_count += 1
        
        return selected_ua.user_agent
    
    def get_next_user_agent(self) -> str:
        """Obtener siguiente User-Agent en rotación secuencial"""
        if self.current_index >= len(self.pool):
            self.current_index = 0
        
        ua = self.pool[self.current_index]
        self.current_index += 1
        
        # Actualizar estadísticas
        ua.last_used = time.time()
        ua.use_count += 1
        
        return ua.user_agent
    
    def get_least_used_user_agent(self) -> str:
        """Obtener User-Agent menos utilizado"""
        least_used = min(self.pool, key=lambda ua: ua.use_count)
        
        # Actualizar estadísticas
        least_used.last_used = time.time()
        least_used.use_count += 1
        
        return least_used.user_agent
    
    def get_academic_user_agent(self) -> str:
        """Obtener User-Agent optimizado para búsquedas académicas"""
        return random.choice(ACADEMIC_USER_AGENTS)
    
    def should_rotate(self) -> bool:
        """Verificar si es tiempo de rotar User-Agent"""
        return time.time() - self.last_rotation_time > self.rotation_interval
    
    def rotate_if_needed(self) -> str:
        """Rotar User-Agent si es necesario"""
        if self.should_rotate():
            self.last_rotation_time = time.time()
            return self.get_random_user_agent()
        return self.get_current_user_agent()
    
    def get_current_user_agent(self) -> str:
        """Obtener User-Agent actual sin rotación"""
        if self.current_index == 0:
            return self.pool[0].user_agent
        return self.pool[self.current_index - 1].user_agent
    
    def get_usage_stats(self) -> Dict[str, any]:
        """Obtener estadísticas de uso"""
        total_uses = sum(ua.use_count for ua in self.pool)
        
        stats = {
            "total_uses": total_uses,
            "unique_agents": len(self.pool),
            "most_used": max(self.pool, key=lambda ua: ua.use_count),
            "least_used": min(self.pool, key=lambda ua: ua.use_count),
            "browser_distribution": {},
            "platform_distribution": {},
        }
        
        # Distribución por navegador
        for browser in BrowserType:
            count = sum(1 for ua in self.pool if ua.browser == browser)
            uses = sum(ua.use_count for ua in self.pool if ua.browser == browser)
            stats["browser_distribution"][browser.value] = {
                "count": count,
                "uses": uses,
                "percentage": (uses / total_uses * 100) if total_uses > 0 else 0
            }
        
        # Distribución por plataforma
        for platform in PlatformType:
            count = sum(1 for ua in self.pool if ua.platform == platform)
            uses = sum(ua.use_count for ua in self.pool if ua.platform == platform)
            stats["platform_distribution"][platform.value] = {
                "count": count,
                "uses": uses,
                "percentage": (uses / total_uses * 100) if total_uses > 0 else 0
            }
        
        return stats
    
    def reset_usage_stats(self):
        """Resetear estadísticas de uso"""
        for ua in self.pool:
            ua.use_count = 0
            ua.last_used = None
        self.current_index = 0
    
    def add_custom_user_agent(self, user_agent: str, 
                            browser: BrowserType,
                            platform: PlatformType,
                            version: str = "unknown",
                            popularity: float = 0.5):
        """Agregar User-Agent personalizado al pool"""
        custom_ua = UserAgentInfo(
            user_agent=user_agent,
            browser=browser,
            platform=platform,
            version=version,
            popularity_score=popularity
        )
        self.pool.append(custom_ua)
    
    def remove_user_agent(self, user_agent: str):
        """Remover User-Agent del pool"""
        self.pool = [ua for ua in self.pool if ua.user_agent != user_agent]
    
    def get_headers_with_user_agent(self, user_agent: str = None) -> Dict[str, str]:
        """Obtener headers completos con User-Agent"""
        if not user_agent:
            user_agent = self.get_random_user_agent()
        
        # Headers comunes que acompañan a User-Agents reales
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }
        
        # Headers específicos por navegador
        if "Chrome" in user_agent:
            headers.update({
                "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
            })
        elif "Firefox" in user_agent:
            headers.update({
                "DNT": "1",
                "Sec-GPC": "1",
            })
        
        return headers

# Instancia global del manager
global_user_agent_manager = UserAgentManager()

# Funciones de conveniencia
def get_random_user_agent(**kwargs) -> str:
    """Función de conveniencia para obtener User-Agent aleatorio"""
    return global_user_agent_manager.get_random_user_agent(**kwargs)

def get_headers_with_random_user_agent(**kwargs) -> Dict[str, str]:
    """Función de conveniencia para obtener headers con User-Agent aleatorio"""
    return global_user_agent_manager.get_headers_with_user_agent(**kwargs)

def get_academic_user_agent() -> str:
    """Función de conveniencia para User-Agent académico"""
    return global_user_agent_manager.get_academic_user_agent()