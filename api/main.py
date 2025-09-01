"""
FastAPI backend for the image encryption system.
Provides REST API endpoints for the React frontend.
"""

import os
import sys
import tempfile
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, Any
import shutil

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import pickle

from aes_cipher import AESCipher
from unified_compression import UnifiedCompressor
from api.file_manager import SecureFileManager

app = FastAPI(title="Image Encryption API", version="1.0.0")

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8081","*"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for file storage  
UPLOAD_DIR = Path("api/uploads")
OUTPUT_DIR = Path("api/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files for downloads
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

# Global instances
compressor = UnifiedCompressor()
file_manager = SecureFileManager(UPLOAD_DIR, OUTPUT_DIR)


def save_image_from_array(array: np.ndarray, path: str, format_ext: str = '.png') -> None:
    """Save numpy array as image file with specified format."""
    img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    
    # Set quality for JPEG format to avoid maximum quality
    if format_ext.lower() in ['.jpg', '.jpeg']:
        img.save(path, quality=95, optimize=True)
    else:
        img.save(path)

def serialize_compressed_data(compressed_data: Dict) -> bytes:
    """Serialize compressed data dictionary to bytes for encryption."""
    return pickle.dumps(compressed_data)

def deserialize_compressed_data(data_bytes: bytes) -> Dict:
    """Deserialize bytes back to compressed data dictionary."""
    return pickle.loads(data_bytes)

def create_encrypted_visualization(encrypted_data: bytes, use_color: bool, output_path: str) -> None:
    """Create visualization image from encrypted data."""
    enc_len = len(encrypted_data)
    
    if use_color:
        # Create RGB visualization
        pad_len = (3 - (enc_len % 3)) % 3
        padded = np.frombuffer(encrypted_data, dtype=np.uint8)
        padded = np.pad(padded, (0, pad_len), "constant", constant_values=0)
        rgb_pixels = padded.reshape(-1, 3)
        side = int(np.ceil(np.sqrt(len(rgb_pixels))))
        pad_pixels = side * side - len(rgb_pixels)
        rgb_pixels = np.pad(
            rgb_pixels, ((0, pad_pixels), (0, 0)), "constant", constant_values=0
        )
        rgb_img = rgb_pixels.reshape(side, side, 3)
        save_image_from_array(rgb_img, output_path)
    else:
        # Create grayscale visualization
        side = int(np.ceil(np.sqrt(enc_len)))
        pad_len = side * side - enc_len
        padded = np.frombuffer(encrypted_data, dtype=np.uint8)
        padded = np.pad(padded, (0, pad_len), "constant", constant_values=0)
        gray_img = padded.reshape(side, side)
        save_image_from_array(gray_img, output_path)

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Image encryption API is running"}

@app.post("/api/encrypt")
async def encrypt_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    key: str = Form(...),
    quality: Optional[int] = Form(75)
):
    """
    Encrypt an image file.
    
    Args:
        image: Image file (.jpg, .jpeg, .png)
        key: 16-character encryption key
        quality: Compression quality 1-100 (default: 75)
    
    Returns:
        JSON with download URLs and statistics
    """
    try:
        # Validate inputs
        if len(key) != 16:
            raise HTTPException(status_code=400, detail="Key must be exactly 16 characters long")
        
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if quality < 1 or quality > 100:
            raise HTTPException(status_code=400, detail="Quality must be between 1 and 100")
        
        # Read and validate file
        file_content = await image.read()
        is_valid, error_msg = file_manager.validate_file(file_content, image.filename or "unknown", image.content_type)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Save uploaded image securely
        upload_path, session_id = file_manager.save_upload(file_content, image.filename or "image")
        
        # Extract original file format for preservation
        original_format = Path(image.filename or "image.png").suffix.lower()
        if original_format not in ['.jpg', '.jpeg', '.png']:
            original_format = '.png'  # Default to PNG for unsupported formats
        
        # Load and process image
        img = Image.open(upload_path)
        original_mode = img.mode
        
        # Convert to supported format
        if original_mode == "L":
            img_np = np.array(img)
            use_color = False
        elif original_mode == "RGB":
            img_np = np.array(img)
            use_color = True
        else:
            # Convert unsupported formats
            if original_mode in ["RGBA", "P", "CMYK", "YCbCr", "LAB", "HSV"]:
                img = img.convert("RGB")
                img_np = np.array(img)
                use_color = True
            else:
                img = img.convert("L")
                img_np = np.array(img)
                use_color = False
        
        # Compress image
        compressed_data = compressor.compress(img_np, quality=quality, use_color=use_color)
        
        # Add original format metadata to compressed data
        compressed_data['original_format'] = original_format
        compressed_data['original_filename'] = image.filename or f"image{original_format}"
        
        # Get compression statistics
        stats = compressor.get_compression_stats(img_np, compressed_data)
        
        # Encrypt compressed data
        key_bytes = key.encode("utf-8")
        cipher = AESCipher(key_bytes)
        compressed_bytes = serialize_compressed_data(compressed_data)
        encrypted_data = cipher.encrypt(compressed_bytes)
        
        # Save encrypted binary file
        encrypted_bin_path = file_manager.save_output(encrypted_data, f"{session_id}_encrypted.bin")
        
        # Create encrypted data visualization
        visualization_path = file_manager.get_output_path(f"{session_id}_visualization.png")
        create_encrypted_visualization(encrypted_data, use_color, str(visualization_path))
        file_manager.file_tracker[str(visualization_path)] = time.time()
        
        # Calculate quality metrics
        compressed_preview = compressor.decompress(compressed_data)
        mse = np.mean((img_np.astype(float) - compressed_preview.astype(float)) ** 2)
        psnr = 20 * np.log10(255) - 10 * np.log10(mse) if mse > 0 else float("inf")
        
        # Files will be automatically cleaned up by the background thread after 30 minutes
        
        return {
            "success": True,
            "session_id": session_id,
            "files": {
                "encrypted_bin": f"/files/{session_id}_encrypted.bin",
                "visualization": f"/files/{session_id}_visualization.png"
            },
            "stats": {
                "original_size": stats['original_size'],
                "compressed_size": stats['compressed_size'],
                "compression_ratio": round(stats['compression_ratio'], 2),
                "mse": round(mse, 2),
                "psnr": round(psnr, 2)
            }
        }
        
    except Exception as e:
        # No immediate cleanup on error - let time-based cleanup handle it
        raise HTTPException(status_code=500, detail=f"Encryption failed: {str(e)}")

@app.post("/api/decrypt")
async def decrypt_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    key: str = Form(...)
):
    """
    Decrypt an encrypted binary file.
    
    Args:
        file: Encrypted .bin file or image file
        key: 16-character decryption key
    
    Returns:
        JSON with download URL and statistics
    """
    try:
        # Validate inputs
        if len(key) != 16:
            raise HTTPException(status_code=400, detail="Key must be exactly 16 characters long")
        
        # Read and validate file
        file_content = await file.read()
        is_valid, error_msg = file_manager.validate_file(file_content, file.filename or "unknown", file.content_type)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Determine file type and handle accordingly
        if file.filename and file.filename.endswith('.bin'):
            # Save uploaded .bin file securely
            upload_path, session_id = file_manager.save_upload(file_content, file.filename)
            encrypted_data = file_content
        
        elif file.content_type and file.content_type.startswith('image/'):
            # Handle image file - assume it's an encrypted visualization
            # In this case, we can't decrypt it as it's just a visualization
            raise HTTPException(
                status_code=400, 
                detail="Cannot decrypt visualization images. Please upload the .bin file instead."
            )
        else:
            raise HTTPException(status_code=400, detail="File must be a .bin file")
        
        # Decrypt data
        key_bytes = key.encode("utf-8")
        cipher = AESCipher(key_bytes)
        
        try:
            decrypted_bytes = cipher.decrypt(encrypted_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Decryption failed. Check your key.")
        
        # Deserialize compressed data
        try:
            compressed_data = deserialize_compressed_data(decrypted_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid encrypted file format")
        
        # Decompress image
        reconstructed_img = compressor.decompress(compressed_data)
        
        # Extract original format from metadata (with fallback to PNG)
        original_format = compressed_data.get('original_format', '.png')
        
        # Save reconstructed image with original format
        output_filename = f"{session_id}_decrypted{original_format}"
        output_path = file_manager.get_output_path(output_filename)
        save_image_from_array(reconstructed_img, str(output_path), original_format)
        # file_manager.file_tracker[str(output_path)] = time.time()
        
        # Files will be automatically cleaned up by the background thread after 30 minutes
        
        return {
            "success": True,
            "session_id": session_id,
            "files": {
                "decrypted_image": f"/files/{session_id}_decrypted{original_format}"
            },
            "stats": {
                "output_size": reconstructed_img.nbytes,
                "image_shape": reconstructed_img.shape
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # No immediate cleanup on error - let time-based cleanup handle it
        raise HTTPException(status_code=500, detail=f"Decryption failed: {str(e)}")

@app.get("/api/download/{filename}")
async def download_file(filename: str, background_tasks: BackgroundTasks):
    """
    Download a generated file.
    
    Args:
        filename: Name of the file to download
    
    Returns:
        File download response
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Schedule cleanup after download
    background_tasks.add_task(file_manager.schedule_cleanup, file_path, 60)
    
    # Determine media type
    if filename.endswith('.bin'):
        media_type = 'application/octet-stream'
    elif filename.endswith(('.png', '.jpg', '.jpeg')):
        media_type = 'image/png' if filename.endswith('.png') else 'image/jpeg'
    else:
        media_type = 'application/octet-stream'
    
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename
    )

# Legacy endpoint for compatibility with existing React code
@app.post("/api/process-image")
async def process_image_legacy(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    key: str = Form(...),
    operation: str = Form(...),
    quality: Optional[int] = Form(75)
):
    """
    Legacy endpoint for backward compatibility with existing React code.
    """
    if operation == "encrypt":
        return await encrypt_image(background_tasks, image, key, quality)
    elif operation == "decrypt":
        return await decrypt_image(background_tasks, image, key)
    else:
        raise HTTPException(status_code=400, detail="Operation must be 'encrypt' or 'decrypt'")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)