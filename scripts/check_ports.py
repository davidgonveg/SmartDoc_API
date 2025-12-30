
import socket
import psutil
import sys
import os
import time

PORTS_TO_CHECK = [6379, 8000, 11434, 8002, 8501]
PORT_NAMES = {
    6379: "Redis",
    8000: "ChromaDB",
    11434: "Ollama",
    8002: "Agent API",
    8501: "Streamlit UI"
}

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def get_process_using_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Use net_connections() instead of info['connections']
            connections = proc.net_connections()
            for conn in connections:
                if conn.laddr.port == port:
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None


def stop_service(service_name):
    """Attempt to stop a system service."""
    import subprocess
    print(f"🔄 Attempting to stop service '{service_name}'...")
    try:
        # Try systemctl first
        subprocess.run(["systemctl", "stop", service_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Try service command as fallback
        subprocess.run(["service", "stop", service_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        return True
    except Exception as e:
        print(f"⚠️ Failed to stop service {service_name}: {e}")
        return False

def kill_process(proc, port_name=None):
    try:
        name = proc.info['name']
        pid = proc.info['pid']
        print(f"⚠️ Killing process {name} (PID {pid})...")
        proc.kill()
        proc.wait(timeout=5)
        print(f"✅ Process terminated.")
        return True
    except psutil.NoSuchProcess:
        print("✅ Process already gone.")
        return True
    except Exception as e:
        print(f"❌ Failed to kill process: {e}")
        return False

def main():
    print("🔍 Checking ports...")
    all_clear = True
    
    # Map ports to likely service names
    SERVICE_MAP = {
        6379: "redis-server",
        11434: "ollama"
    }
    
    for port in PORTS_TO_CHECK:
        if is_port_in_use(port):
            port_name = PORT_NAMES.get(port, 'Unknown')
            print(f"⚠️ Port {port} ({port_name}) is in use.")
            
            # Retry loop for stubborn processes
            max_retries = 3
            for attempt in range(max_retries):
                proc = get_process_using_port(port)
                if not proc:
                    print(f"✅ Port {port} released.")
                    break
                
                print(f"   Occupied by: {proc.info['name']} (PID {proc.info['pid']})")
                
                # If we killed it before and it's back (different PID or same), it might be a service
                if attempt > 0:
                     print(f"   It seems to be restarting. Checking for services...")
                     if port in SERVICE_MAP:
                         stop_service(SERVICE_MAP[port])

                kill_process(proc)
                time.sleep(2) # Wait for release
                
            if is_port_in_use(port):
                 print(f"❌ Port {port} is still in use after {max_retries} attempts. Please stop the '{port_name}' service manually.")
                 all_clear = False
            else:
                 print(f"✅ Port {port} validated free.")
        else:
            print(f"✅ Port {port} is free.")
            
    if all_clear:
        print("✅ All ports are ready.")
        sys.exit(0)
    else:
        print("❌ Some ports are still occupied. Please check manually.")
        sys.exit(1)

if __name__ == "__main__":
    main()
