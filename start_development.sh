#!/bin/bash

echo "🚀 Starting Image Encryption System"
echo "===================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment and install dependencies
echo "📦 Installing/updating dependencies..."
source venv/bin/activate
cd api
pip install -r requirements.txt > /dev/null 2>&1
cd ..

echo "✅ Dependencies installed"
echo ""

# Start FastAPI server
echo "🔧 Starting FastAPI backend server..."
echo "📍 Backend will be available at: http://localhost:8000"
echo "📚 API docs available at: http://localhost:8000/docs"
echo ""

cd api
python -c "
import uvicorn
from main import app
print('🎯 FastAPI server starting on http://localhost:8000')
print('Press Ctrl+C to stop the server')
print('=' * 50)
uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)
"