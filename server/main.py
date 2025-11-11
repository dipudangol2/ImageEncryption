"""
FastAPI backend for the image encryption system.
Provides REST API endpoints for the React frontend.
"""

import sys
import os
import time
import hashlib
import uuid
from pathlib import Path
from typing import Optional, Dict, Any



# Add parent directory to path to import our modules
# sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import pickle

from services.AES import AESCipher
from services.DCT import DCTCompressor
from services.file_manager import FileManager
from services.performance_analyzer import PerformanceAnalyzer

app = FastAPI(title="Image Encryption API", version="1.0.0")

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8081",
        "http://localhost",
        
    ],  # React dev servers
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

# Global instances - Using Manual DCT with Quantization (REAL COMPRESSION!)
compressor = DCTCompressor()
file_manager = FileManager(UPLOAD_DIR, OUTPUT_DIR)
performance_analyzer = PerformanceAnalyzer()


def derive_aes_key(password: str) -> bytes:
    """
    Derive a 16-byte AES key from any-length password using SHA-256.

    Args:
        password: Password of any length

    Returns:
        16-byte key for AES-128 encryption
    """
    # Create SHA-256 hash of the password
    sha256_hash = hashlib.sha256(password.encode("utf-8")).digest()
    # Return first 16 bytes for AES-128
    return sha256_hash[:16]


def serialize_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: serialize_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_numpy_types(item) for item in obj]
    elif obj is None:
        return None
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        # Handle other types by converting to string as fallback
        return str(obj)


def save_image_from_array(
    array: np.ndarray, path: str, format_ext: str = ".png"
) -> None:
    """Save numpy array as image file with specified format."""
    img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

    # Set quality for JPEG format to avoid maximum quality
    if format_ext.lower() in [".jpg", ".jpeg"]:
        img.save(path, quality=95, optimize=True)
    else:
        img.save(path)


def serialize_compressed_data(compressed_data: Dict) -> bytes:
    """Serialize compressed data dictionary to bytes for encryption."""
    return pickle.dumps(compressed_data)


def deserialize_compressed_data(data_bytes: bytes) -> Dict:
    """Deserialize bytes back to compressed data dictionary."""
    return pickle.loads(data_bytes)


def create_encrypted_visualization(
    encrypted_data: bytes, use_color: bool, output_path: str
) -> None:
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
    quality: Optional[int] = Form(75),
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
        # Reset performance analyzer for new operation
        performance_analyzer.reset_timing()

        # Validate inputs - password can be any length
        if len(key.strip()) == 0:
            raise HTTPException(status_code=400, detail="Password cannot be empty")

        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        if quality < 1 or quality > 100:
            raise HTTPException(
                status_code=400, detail="Quality must be between 1 and 100"
            )

        # Read and validate file
        file_content = await image.read()
        original_file_size = image.size  # Get original file size in bytes

        is_valid, error_msg = file_manager.validate_file(
            file_content, image.filename or "unknown", image.content_type
        )

        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # Save uploaded image securely
        upload_path, session_id = file_manager.save_upload(
            file_content, image.filename or "image"
        )

        # Extract original file format for preservation
        original_format = Path(image.filename or "image.png").suffix.lower()
        if original_format not in [".jpg", ".jpeg", ".png"]:
            original_format = ".png"  # Default to PNG for unsupported formats

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

        # Set quality based on file format
        if original_format.lower() in [".jpg", ".jpeg"]:
            # JPG files get lower quality for better compression
            compression_quality = 30
        elif original_format.lower() == ".png":
            # PNG files get higher quality
            compression_quality = 60
        else:
            # Other formats (including AVIF) use default quality
            compression_quality = quality

        # Compress image with timing
        with performance_analyzer.time_operation("compression"):
            compressed_data = compressor.compress(
                img_np, quality=compression_quality, use_color=use_color
            )

        # Add original format metadata to compressed data
        compressed_data["original_format"] = original_format
        compressed_data["original_filename"] = (
            image.filename or f"image{original_format}"
        )
        # Store the actual compression quality used (for stats reporting)
        compressed_data["actual_quality_used"] = compression_quality

        # Encrypt compressed data with timing
        with performance_analyzer.time_operation("encryption"):
            key_bytes = derive_aes_key(key)
            cipher = AESCipher(key_bytes)
            compressed_bytes = serialize_compressed_data(compressed_data)
            encrypted_data = cipher.encrypt(compressed_bytes)

        # Save encrypted binary file
        encrypted_bin_path = file_manager.save_output(
            encrypted_data, f"{session_id}_encrypted.bin"
        )

        # Create encrypted data visualization with timing
        with performance_analyzer.time_operation("visualization"):
            visualization_path = file_manager.get_output_path(
                f"{session_id}_visualization.png"
            )
            create_encrypted_visualization(
                encrypted_data, use_color, str(visualization_path)
            )
            file_manager.file_tracker[str(visualization_path)] = time.time()

        # Decompress for quality analysis
        with performance_analyzer.time_operation("decompression"):
            reconstructed_img = compressor.decompress(compressed_data)

        # Load encrypted visualization for analysis
        encrypted_visualization = np.array(Image.open(visualization_path))

        # Generate comprehensive performance report
        performance_report = performance_analyzer.generate_performance_report(
            img_np, compressed_data, reconstructed_img, encrypted_visualization
        )

        # Get basic compression statistics
        basic_stats = compressor.get_compression_stats(img_np, compressed_data)

        # Serialize numpy types for JSON response
        performance_report_serialized = serialize_numpy_types(performance_report)

        # Calculate total processing time
        timing_data = performance_analyzer.get_timing_report()
        total_time = timing_data.get("total_time", 0)

        # Build comprehensive stats response with original file size
        def format_size(size_bytes):
            if size_bytes < 1024:
                return f"{size_bytes} bytes"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"

        # Calculate compression ratio using original file size vs our compressed size
        file_compression_ratio = (
            original_file_size / basic_stats["compressed_size"]
            if basic_stats["compressed_size"] > 0
            else 1.0
        )
        file_space_saved = (
            ((original_file_size - basic_stats["compressed_size"]) / original_file_size)
            * 100
            if original_file_size > 0
            else 0
        )

        comprehensive_stats = {
            # File size comparison (original file vs our compression)
            "original_size": format_size(original_file_size),
            "compressed_size": basic_stats["compressed_size_display"],
            "compression_ratio": round(float(file_compression_ratio), 2),
            "space_saved_percent": round(float(file_space_saved), 1),
            # Image dimension metrics
            "image_width": int(img_np.shape[1]),
            "image_height": int(img_np.shape[0]),
            "image_dimensions": f"{img_np.shape[1]} × {img_np.shape[0]}",
            "image_channels": int(img_np.shape[2] if len(img_np.shape) > 2 else 1),
            "total_pixels": int(img_np.shape[0] * img_np.shape[1]),
            "image_format": original_format,
            # Actual compression quality used
            "quality": compression_quality,
            # File size metrics in bytes for precision
            "original_size_bytes": int(original_file_size),
            "compressed_size_bytes": int(basic_stats["compressed_size"]),
            # Array compression metrics (for technical analysis)
            "array_size_bytes": int(basic_stats["original_size"]),
            "array_compression_ratio": round(
                float(basic_stats["compression_ratio"]), 2
            ),
            "array_space_saved_percent": round(
                float(basic_stats["space_saved_percent"]), 1
            ),
            # Quality metrics
            "mse": round(
                float(performance_report_serialized["quality_metrics"]["mse"]), 2
            ),
            "psnr": (
                round(
                    float(performance_report_serialized["quality_metrics"]["psnr"]), 2
                )
                if performance_report_serialized["quality_metrics"]["psnr"]
                != float("inf")
                else 999.99
            ),
            # Encryption security metrics
            "npcr": round(
                float(performance_report_serialized["encryption_metrics"]["npcr"]), 2
            ),
            "uaci": round(
                float(performance_report_serialized["encryption_metrics"]["uaci"]), 2
            ),
            "encryption_strength": performance_report_serialized["encryption_metrics"][
                "strength_evaluation"
            ]["strength_level"],
            # Compression efficiency
            "bits_per_pixel": round(
                float(
                    performance_report_serialized["compression_metrics"][
                        "bits_per_pixel"
                    ]
                ),
                3,
            ),
            "compression_efficiency": round(
                float(
                    performance_report_serialized["compression_metrics"][
                        "compression_efficiency"
                    ]
                ),
                2,
            ),
            # Timing analysis
            "total_processing_time": round(float(total_time), 3),
            "compression_time": round(
                float(timing_data["operations"].get("compression", 0)), 3
            ),
            "encryption_time": round(
                float(timing_data["operations"].get("encryption", 0)), 3
            ),
            "decompression_time": round(
                float(timing_data["operations"].get("decompression", 0)), 3
            ),
            "visualization_time": round(
                float(timing_data["operations"].get("visualization", 0)), 3
            ),
            # Entropy analysis
            "original_entropy": round(
                float(
                    performance_report_serialized["histogram_analysis"]["original"][
                        "entropy"
                    ]
                    if performance_report_serialized["histogram_analysis"]["original"][
                        "type"
                    ]
                    == "grayscale"
                    else performance_report_serialized["histogram_analysis"][
                        "original"
                    ]["channels"]["red"]["entropy"]
                ),
                2,
            ),
            "encrypted_entropy": round(
                float(
                    performance_report_serialized["histogram_analysis"]["encrypted"][
                        "entropy"
                    ]
                    if performance_report_serialized["histogram_analysis"]["encrypted"][
                        "type"
                    ]
                    == "grayscale"
                    else performance_report_serialized["histogram_analysis"][
                        "encrypted"
                    ]["channels"]["red"]["entropy"]
                ),
                2,
            ),
            # Summary ratings
            "overall_quality": performance_report_serialized["summary"][
                "overall_quality"
            ],
            "encryption_security": performance_report_serialized["summary"][
                "encryption_security"
            ],
            "compression_efficiency_rating": performance_report_serialized["summary"][
                "compression_efficiency"
            ],
            # Additional detailed metrics for advanced users
            "detailed_metrics": performance_report_serialized,
        }

        # Files will be automatically cleaned up by the background thread after 30 minutes

        return {
            "success": True,
            "session_id": session_id,
            "files": {
                "encrypted_bin": f"/files/{session_id}_encrypted.bin",
                "visualization": f"/files/{session_id}_visualization.png",
            },
            "stats": comprehensive_stats,
        }

    except Exception as e:
        # No immediate cleanup on error - let time-based cleanup handle it
        raise HTTPException(status_code=500, detail=f"Encryption failed: {str(e)}")


@app.post("/api/decrypt")
async def decrypt_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    key: str = Form(...),
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
        # Reset performance analyzer for new operation
        performance_analyzer.reset_timing()

        # Validate inputs - password can be any length
        if len(key.strip()) == 0:
            raise HTTPException(status_code=400, detail="Password cannot be empty")

        # Read and validate file
        file_content = await file.read()
        is_valid, error_msg = file_manager.validate_file(
            file_content, file.filename or "unknown", file.content_type
        )

        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # Determine file type and handle accordingly
        if file.filename and file.filename.endswith(".bin"):
            # Save uploaded .bin file securely
            upload_path, session_id = file_manager.save_upload(
                file_content, file.filename
            )
            encrypted_data = file_content

        elif file.content_type and file.content_type.startswith("image/"):
            # Handle image file - assume it's an encrypted visualization
            # In this case, we can't decrypt it as it's just a visualization
            raise HTTPException(
                status_code=400,
                detail="Cannot decrypt visualization images. Please upload the .bin file instead.",
            )
        else:
            raise HTTPException(status_code=400, detail="File must be a .bin file")

        # Decrypt data with timing
        with performance_analyzer.time_operation("decryption"):
            key_bytes = derive_aes_key(key)
            cipher = AESCipher(key_bytes)
            try:
                decrypted_bytes = cipher.decrypt(encrypted_data)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail="Decryption failed. Check your key."
                )

        # Deserialize compressed data
        try:
            compressed_data = deserialize_compressed_data(decrypted_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid encrypted file format")

        # Decompress image with timing
        with performance_analyzer.time_operation("decompression"):
            reconstructed_img = compressor.decompress(compressed_data)

        # Extract original format from metadata (with fallback to PNG)
        # original_format = ".jpg"
        original_format = compressed_data.get("original_format", ".jpg")

        # Save reconstructed image with original format
        output_filename = f"{session_id}_decrypted{original_format}"
        output_path = file_manager.get_output_path(output_filename)
        save_image_from_array(reconstructed_img, str(output_path), original_format)

        # Generate comprehensive histogram analysis for decrypted image
        with performance_analyzer.time_operation("analysis"):
            decrypted_histogram = performance_analyzer.analyze_histogram(reconstructed_img)

        # Get timing report
        timing_data = performance_analyzer.get_timing_report()
        total_time = timing_data.get("total_time", 0)

        # Serialize histogram data for JSON response
        decrypted_histogram_serialized = serialize_numpy_types(decrypted_histogram)

        # Calculate entropy for summary
        decrypted_entropy = (
            decrypted_histogram_serialized["entropy"]
            if decrypted_histogram_serialized["type"] == "grayscale"
            else decrypted_histogram_serialized["channels"]["red"]["entropy"]
        )

        # Build comprehensive stats for decryption
        comprehensive_stats = {
            # Basic file information
            "output_size": int(reconstructed_img.nbytes),
            "image_shape": list(reconstructed_img.shape),
            "original_format": original_format,
            "quality": compressed_data.get(
                "actual_quality_used", compressed_data.get("quality", 75)
            ),
            "output_size_display": (
                f"{reconstructed_img.nbytes / 1024:.1f} KB"
                if reconstructed_img.nbytes >= 1024
                else f"{reconstructed_img.nbytes} bytes"
            ),
            # Image dimension metrics
            "image_width": int(reconstructed_img.shape[1]),
            "image_height": int(reconstructed_img.shape[0]),
            "image_dimensions": f"{reconstructed_img.shape[1]} × {reconstructed_img.shape[0]}",
            "image_channels": int(
                reconstructed_img.shape[2] if len(reconstructed_img.shape) > 2 else 1
            ),
            "total_pixels": int(
                reconstructed_img.shape[0] * reconstructed_img.shape[1]
            ),
            # Comprehensive timing analysis for decryption
            "total_processing_time": round(float(total_time), 3),
            "decryption_time": round(
                float(timing_data["operations"].get("decryption", 0)), 3
            ),
            "decompression_time": round(
                float(timing_data["operations"].get("decompression", 0)), 3
            ),
            "analysis_time": round(
                float(timing_data["operations"].get("analysis", 0)), 3
            ),
            # Initialize other timing fields to 0 for consistency
            "compression_time": 0,
            "encryption_time": 0,
            "visualization_time": 0,
            # Image analysis
            "mean_intensity": round(float(np.mean(reconstructed_img)), 2),
            "std_intensity": round(float(np.std(reconstructed_img)), 2),
            "dynamic_range": round(
                float(np.max(reconstructed_img) - np.min(reconstructed_img)), 2
            ),
            "unique_values": int(len(np.unique(reconstructed_img))),
            # Entropy analysis for decrypted image
            "decrypted_entropy": round(float(decrypted_entropy), 2),
            # File size efficiency
            "bytes_per_pixel": round(
                float(
                    reconstructed_img.nbytes
                    / (reconstructed_img.shape[0] * reconstructed_img.shape[1])
                ),
                2,
            ),
            # Detailed histogram analysis for comprehensive results page
            "detailed_metrics": {
                "histogram_analysis": {
                    "decrypted": decrypted_histogram_serialized
                },
                "timing_analysis": timing_data,
                "image_properties": {
                    "format": original_format,
                    "quality": compressed_data.get("actual_quality_used", compressed_data.get("quality", 75)),
                    "dimensions": {
                        "width": int(reconstructed_img.shape[1]),
                        "height": int(reconstructed_img.shape[0]),
                        "channels": int(reconstructed_img.shape[2] if len(reconstructed_img.shape) > 2 else 1)
                    }
                }
            }
        }

        # Files will be automatically cleaned up by the background thread after 30 minutes

        return {
            "success": True,
            "session_id": session_id,
            "files": {
                "decrypted_image": f"/files/{session_id}_decrypted{original_format}"
            },
            "stats": comprehensive_stats,
        }

    except HTTPException:
        raise
    except Exception as e:
        # No immediate cleanup on error - let time-based cleanup handle it
        raise HTTPException(status_code=500, detail=f"Decryption failed: {str(e)}")


@app.post("/api/comprehensive-analysis")
async def comprehensive_analysis(
    background_tasks: BackgroundTasks,
    encrypted_file: UploadFile = File(...),
    key: str = Form(...),
    original_session_id: str = Form(...),
):
    """
    Perform comprehensive three-way analysis (original, encrypted, decrypted).
    
    This endpoint is designed for the results page to get complete comparison data.
    It requires the original image session data and performs decryption analysis.
    
    Args:
        encrypted_file: The encrypted .bin file
        key: Decryption password
        original_session_id: Session ID from the original encryption operation
        
    Returns:
        Complete three-way analysis data for results page
    """
    try:
        # Reset performance analyzer for new operation
        performance_analyzer.reset_timing()
        
        # Validate inputs
        if len(key.strip()) == 0:
            raise HTTPException(status_code=400, detail="Password cannot be empty")
            
        if not original_session_id:
            raise HTTPException(status_code=400, detail="Original session ID required")
        
        # Check if original files exist
        original_upload_path = None
        original_visualization_path = None
        
        # Try to find original files by session ID
        try:
            upload_files = list(UPLOAD_DIR.glob(f"{original_session_id}_upload.*"))
            for file_path in upload_files:
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    original_upload_path = file_path
                    break
                    
            viz_files = list(OUTPUT_DIR.glob(f"{original_session_id}_visualization.*"))
            for file_path in viz_files:
                if file_path.suffix.lower() == '.png':
                    original_visualization_path = file_path
                    break
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error searching for session files: {str(e)}"
            )
        
        if not original_upload_path:
            raise HTTPException(
                status_code=404,
                detail=f"Original image file not found for session {original_session_id}. It may have been cleaned up."
            )
            
        if not original_visualization_path:
            raise HTTPException(
                status_code=404,
                detail=f"Encrypted visualization not found for session {original_session_id}. It may have been cleaned up."
            )
        
        # Read and validate encrypted file
        encrypted_content = await encrypted_file.read()
        is_valid, error_msg = file_manager.validate_file(
            encrypted_content, encrypted_file.filename or "unknown", encrypted_file.content_type
        )
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
            
        if not (encrypted_file.filename and encrypted_file.filename.endswith(".bin")):
            raise HTTPException(status_code=400, detail="File must be a .bin file")
        
        # Decrypt data with timing
        with performance_analyzer.time_operation("decryption"):
            key_bytes = derive_aes_key(key)
            cipher = AESCipher(key_bytes)
            try:
                decrypted_bytes = cipher.decrypt(encrypted_content)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail="Decryption failed. Check your password."
                )
        
        # Deserialize compressed data
        try:
            compressed_data = deserialize_compressed_data(decrypted_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid encrypted file format")
        
        # Decompress image with timing
        with performance_analyzer.time_operation("decompression"):
            reconstructed_img = compressor.decompress(compressed_data)
        
        # Load original image and encrypted visualization
        with performance_analyzer.time_operation("analysis"):
            try:
                # Load original image
                original_pil = Image.open(original_upload_path)
                
                # Convert to match reconstructed image format (same as encryption does)
                if len(reconstructed_img.shape) == 2:
                    # Reconstructed is grayscale
                    if original_pil.mode != "L":
                        original_pil = original_pil.convert("L")
                else:
                    # Reconstructed is RGB
                    if original_pil.mode != "RGB":
                        original_pil = original_pil.convert("RGB")
                
                original_img = np.array(original_pil)
                encrypted_visualization = np.array(Image.open(original_visualization_path))
                
                # Now shapes should match for PSNR calculation
                psnr = performance_analyzer.calculate_psnr(original_img, reconstructed_img)
                mse = ((255.0 / (10 ** (psnr / 20))) ** 2) if psnr != float('inf') else 0.0
                
                # For NPCR/UACI, resize encrypted visualization to match original for proper comparison
                encrypted_resized = performance_analyzer._resize_to_match(encrypted_visualization, original_img.shape)
                npcr = performance_analyzer.calculate_npcr(original_img, encrypted_resized)
                uaci = performance_analyzer.calculate_uaci(original_img, encrypted_resized)
                encryption_strength = performance_analyzer.evaluate_encryption_strength(npcr, uaci)
                
                # Analyze histograms for all three images
                original_histogram = performance_analyzer.analyze_histogram(original_img)
                encrypted_histogram = performance_analyzer.analyze_histogram(encrypted_visualization)
                decrypted_histogram = performance_analyzer.analyze_histogram(reconstructed_img)
                
                # Compression efficiency
                compression_metrics = performance_analyzer.analyze_compression_efficiency(original_img, compressed_data)
                
                # Build comprehensive performance report manually
                performance_report = {
                    'quality_metrics': {
                        'psnr': psnr,
                        'mse': mse
                    },
                    'encryption_metrics': {
                        'npcr': npcr,
                        'uaci': uaci,
                        'strength_evaluation': encryption_strength
                    },
                    'compression_metrics': compression_metrics,
                    'histogram_analysis': {
                        'original': original_histogram,
                        'encrypted': encrypted_histogram,
                        'decrypted': decrypted_histogram
                    },
                    'timing_analysis': {},  # Will be filled below
                    'summary': {
                        'overall_quality': 'Excellent' if psnr > 40 else 'Good' if psnr > 30 else 'Fair' if psnr > 20 else 'Poor',
                        'encryption_security': encryption_strength['strength_level'],
                        'compression_efficiency': 'High' if compression_metrics['compression_ratio'] > 5 else 'Medium' if compression_metrics['compression_ratio'] > 2 else 'Low'
                    }
                }
                
            except Exception as e:
                print(f"Analysis error: {str(e)}")
                import traceback
                traceback.print_exc()
                # More detailed error logging
                print(f"Original image shape: {original_img.shape if 'original_img' in locals() else 'Not loaded'}")
                print(f"Reconstructed image shape: {reconstructed_img.shape if 'reconstructed_img' in locals() else 'Not loaded'}")
                print(f"Encrypted visualization shape: {encrypted_visualization.shape if 'encrypted_visualization' in locals() else 'Not loaded'}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis failed: {str(e)}"
                )
        
        # Save decrypted image for results page
        decrypted_session_id = str(uuid.uuid4())
        original_format = compressed_data.get("original_format", ".png")
        output_filename = f"{decrypted_session_id}_decrypted{original_format}"
        output_path = file_manager.get_output_path(output_filename)
        save_image_from_array(reconstructed_img, str(output_path), original_format)
        
        # Serialize data for JSON response
        performance_report_serialized = serialize_numpy_types(performance_report)
        timing_data = performance_analyzer.get_timing_report()
        total_time = timing_data.get("total_time", 0)
        
        # Build file URLs for three-way comparison
        # Note: The upload files are in UPLOAD_DIR, not OUTPUT_DIR, so they aren't served by /files/
        # We need to copy the original file to OUTPUT_DIR for serving
        original_output_filename = f"{original_session_id}_original{Path(original_upload_path).suffix}"
        original_output_path = file_manager.get_output_path(original_output_filename)
        
        # Copy original file to output directory if it doesn't exist there
        if not original_output_path.exists():
            import shutil
            shutil.copy2(original_upload_path, original_output_path)
            file_manager.file_tracker[str(original_output_path)] = time.time()
        
        file_urls = {
            "original": f"/files/{original_output_filename}",
            "encrypted_visualization": f"/files/{original_session_id}_visualization.png",
            "decrypted": f"/files/{decrypted_session_id}_decrypted{original_format}"
        }
        
        # Calculate comprehensive comparison metrics
        comparison_metrics = {
            "quality_comparison": {
                "psnr_original_vs_decrypted": performance_report_serialized["quality_metrics"]["psnr"],
                "mse_original_vs_decrypted": performance_report_serialized["quality_metrics"]["mse"]
            },
            "encryption_security": {
                "npcr_original_vs_encrypted": performance_report_serialized["encryption_metrics"]["npcr"],
                "uaci_original_vs_encrypted": performance_report_serialized["encryption_metrics"]["uaci"],
                "encryption_strength": performance_report_serialized["encryption_metrics"]["strength_evaluation"]["strength_level"]
            },
            "timing_breakdown": {
                "total_time": round(float(total_time), 3),
                "decryption_time": round(float(timing_data["operations"].get("decryption", 0)), 3),
                "decompression_time": round(float(timing_data["operations"].get("decompression", 0)), 3),
                "analysis_time": round(float(timing_data["operations"].get("analysis", 0)), 3)
            },
            "compression_analysis": performance_report_serialized["compression_metrics"]
        }
        
        return {
            "success": True,
            "session_ids": {
                "original": original_session_id,
                "decrypted": decrypted_session_id
            },
            "files": file_urls,
            "comprehensive_analysis": {
                "histogram_data": {
                    "original": performance_report_serialized["histogram_analysis"]["original"],
                    "encrypted": performance_report_serialized["histogram_analysis"]["encrypted"],
                    "decrypted": performance_report_serialized["histogram_analysis"]["decrypted"]
                },
                "comparison_metrics": comparison_metrics,
                "detailed_metrics": performance_report_serialized,
                "summary": performance_report_serialized["summary"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


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
    background_tasks.add_task(file_manager.schedule_cleanup, file_path, 90)

    # Determine media type
    if filename.endswith(".bin"):
        media_type = "application/octet-stream"
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        media_type = "image/png" if filename.endswith(".png") else "image/jpeg"
    else:
        media_type = "application/octet-stream"

    return FileResponse(file_path, media_type=media_type, filename=filename)


# Legacy endpoint for compatibility with existing React code
@app.post("/api/process-image")
async def process_image_legacy(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    key: str = Form(...),
    operation: str = Form(...),
    quality: Optional[int] = Form(75),
):
    """
    Legacy endpoint for backward compatibility with existing React code.
    """
    if operation == "encrypt":
        return await encrypt_image(background_tasks, image, key, quality)
    elif operation == "decrypt":
        return await decrypt_image(background_tasks, image, key)
    else:
        raise HTTPException(
            status_code=400, detail="Operation must be 'encrypt' or 'decrypt'"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
