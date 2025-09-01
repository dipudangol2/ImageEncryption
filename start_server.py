#!/usr/bin/env python3
"""
Startup script for the Image Encryption FastAPI server.
"""

import os
import sys
from pathlib import Path

# Ensure we're in the correct directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Add current directory to Python path
sys.path.insert(0, str(script_dir))

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Image Encryption API Server...")
    print("📁 API will be available at: http://localhost:8000")
    print("📱 React frontend should run on: http://localhost:5173")
    print("🔗 API docs available at: http://localhost:8000/docs")
    print("\n⚠️  Make sure to install FastAPI dependencies:")
    print("   cd api && pip install -r requirements.txt")
    print("\n" + "="*50)
    
    # Start the server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(script_dir)]
    )