#!/usr/bin/env python3
"""
Simple debug test for database import
"""

import sys
import os
from pathlib import Path

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")
print(f"App path: {cluster_app_path}")
print(f"App path exists: {cluster_app_path.exists()}")

try:
    import database
    print("✅ database module imported successfully")
    print(f"Database module file: {database.__file__}")
    print(f"Database module attributes: {dir(database)}")
    
    if hasattr(database, 'ClusterDatabase'):
        print("✅ ClusterDatabase class found")
        from database import ClusterDatabase
        print("✅ ClusterDatabase imported successfully")
    else:
        print("❌ ClusterDatabase class not found in database module")
        
except Exception as e:
    print(f"❌ Failed to import database: {e}")
    import traceback
    traceback.print_exc()
