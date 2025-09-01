# Image Encryption System

A complete image encryption system with DCT compression and AES-128 encryption, featuring a React frontend and FastAPI backend.

## Features

- **Image Compression**: DCT-based compression with quality control
- **AES-128 Encryption**: Secure encryption with 16-character keys
- **Histogram Analysis**: Encrypted data visualization for analysis
- **Web Interface**: Modern React frontend with drag-drop support
- **File Support**: Images (.jpg, .jpeg, .png) and encrypted .bin files
- **Secure API**: FastAPI backend with file validation and cleanup

## Quick Start

### 1. Install Backend Dependencies
```bash
cd api
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies
```bash
cd pixel-guard-react
npm install
```

### 3. Start the System
```bash
# Terminal 1: Start FastAPI backend
python start_server.py

# Terminal 2: Start React frontend
cd pixel-guard-react
npm run dev
```

### 4. Access the Application
- Frontend: http://localhost:5173
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## API Endpoints

- `POST /api/encrypt` - Encrypt image files
- `POST /api/decrypt` - Decrypt .bin files  
- `GET /api/download/{filename}` - Download generated files
- `GET /api/health` - Health check

## Usage

1. **Encrypt**: Upload image → Enter 16-char key → Get encrypted .bin + visualization
2. **Decrypt**: Upload .bin file → Enter same key → Get original image

## File Output

- `encrypted_data.bin` - Encrypted binary for secure storage
- `encrypted_visualization.png` - For histogram analysis
- `decrypted_image.png` - Reconstructed original image

## Command Line Usage

You can also use the standalone scripts:
```bash
python main2.py image.jpg "your16charkey123" --quality 75
```
