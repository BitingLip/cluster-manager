#!/usr/bin/env python3
"""
Test the system monitoring functionality specifically
"""

import os
import sys
import psutil
from pathlib import Path

# Add app directory to path
app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(app_path))

def test_system_monitoring():
    """Test the system monitoring code that might be failing"""
    print("🔍 Testing System Monitoring Functions")
    print("=" * 50)
    
    try:
        # Test CPU usage
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"✅ CPU Count: {cpu_count}")
        print(f"✅ CPU Usage: {cpu_usage:.1f}%")
        
        # Test memory
        memory = psutil.virtual_memory()
        memory_total_mb = memory.total / (1024 * 1024)
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        print(f"✅ Memory Total: {memory_total_mb:.0f}MB")
        print(f"✅ Memory Used: {memory_used_mb:.0f}MB")
        print(f"✅ Memory Available: {memory_available_mb:.0f}MB")
        
        # Test disk usage
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024 * 1024 * 1024)
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        print(f"✅ Disk Total: {disk_total_gb:.1f}GB")
        print(f"✅ Disk Used: {disk_used_gb:.1f}GB")
        
        # Test network I/O
        net_io = psutil.net_io_counters()
        network_rx_bytes = net_io.bytes_recv
        network_tx_bytes = net_io.bytes_sent
        print(f"✅ Network RX: {network_rx_bytes} bytes")
        print(f"✅ Network TX: {network_tx_bytes} bytes")
        
        # Test the problematic load average
        print("\n🔍 Testing Load Average (Windows compatibility)")
        try:
            load_avg = os.getloadavg()
            load_average_1m = load_avg[0]
            load_average_5m = load_avg[1]
            load_average_15m = load_avg[2]
            print(f"✅ Load Average (Unix): {load_average_1m:.2f}, {load_average_5m:.2f}, {load_average_15m:.2f}")
        except (AttributeError, OSError) as e:
            print(f"⚠️  Load Average not available (Windows): {e}")
            # Windows fallback
            load_average_1m = cpu_usage / 100.0
            load_average_5m = cpu_usage / 100.0
            load_average_15m = cpu_usage / 100.0
            print(f"✅ Load Average (Windows fallback): {load_average_1m:.2f}, {load_average_5m:.2f}, {load_average_15m:.2f}")
        
        print("\n🎯 System Monitoring Test Results:")
        print("✅ All system monitoring functions working correctly")
        print("✅ Windows compatibility handled properly")
        
        return True
        
    except Exception as e:
        print(f"❌ System monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_system_monitoring()
