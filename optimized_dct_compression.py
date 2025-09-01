import numpy as np
from scipy.fft import dct, idct
import pickle
from typing import Tuple, Dict, Optional, Union, List
from collections import Counter

class SimplifiedDCTCompressor:
    """
    Reliable DCT-based image compressor with basic compression optimizations.
    """
    
    def __init__(self):
        # Pre-computed zigzag indices for 8x8 blocks
        self.zigzag_indices = self._generate_zigzag_indices()
        
        # Standard JPEG quantization matrices
        self.luminance_quant = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)
        
        # Chrominance quantization matrix (more aggressive for color channels)
        self.chrominance_quant = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=np.float32)
    
    def _generate_zigzag_indices(self) -> np.ndarray:
        """Generate zigzag scan indices for 8x8 block."""
        zigzag_order = [
            (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
            (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
            (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
            (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
            (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
            (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
            (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
            (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
        ]
        return np.array([i * 8 + j for i, j in zigzag_order])
    
    def _rgb_to_ycbcr(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB to YCbCr color space for better compression."""
        # Standard RGB to YCbCr conversion matrix
        transform_matrix = np.array([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ], dtype=np.float32)
        
        # Reshape for matrix multiplication
        h, w, c = rgb_image.shape
        rgb_flat = rgb_image.reshape(-1, 3).astype(np.float32)
        
        # Apply transformation
        ycbcr_flat = np.dot(rgb_flat, transform_matrix.T)
        
        # Add offsets for Cb and Cr channels
        ycbcr_flat[:, 1:] += 128
        
        return ycbcr_flat.reshape(h, w, c)
    
    def _ycbcr_to_rgb(self, ycbcr_image: np.ndarray) -> np.ndarray:
        """Convert YCbCr back to RGB."""
        # Inverse transformation matrix
        inv_transform_matrix = np.array([
            [1.0, 0.0, 1.402],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.772, 0.0]
        ], dtype=np.float32)
        
        h, w, c = ycbcr_image.shape
        ycbcr_flat = ycbcr_image.reshape(-1, 3).astype(np.float32)
        
        # Remove offsets from Cb and Cr channels
        ycbcr_flat[:, 1:] -= 128
        
        # Apply inverse transformation
        rgb_flat = np.dot(ycbcr_flat, inv_transform_matrix.T)
        
        return np.clip(rgb_flat.reshape(h, w, c), 0, 255)
    
    def _fast_dct_2d(self, blocks: np.ndarray) -> np.ndarray:
        """Vectorized 2D DCT using scipy."""
        return dct(dct(blocks, axis=2, norm='ortho'), axis=1, norm='ortho')
    
    def _fast_idct_2d(self, dct_blocks: np.ndarray) -> np.ndarray:
        """Vectorized 2D inverse DCT using scipy."""
        return idct(idct(dct_blocks, axis=2, norm='ortho'), axis=1, norm='ortho')
    
    def _extract_blocks(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract 8x8 blocks from image with padding."""
        h, w = image.shape
        padded_h = ((h + 7) // 8) * 8
        padded_w = ((w + 7) // 8) * 8
        
        # Pad image
        padded_image = np.zeros((padded_h, padded_w), dtype=np.float32)
        padded_image[:h, :w] = image.astype(np.float32)
        
        # Extract all blocks at once
        blocks = padded_image.reshape(padded_h // 8, 8, padded_w // 8, 8)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, 8, 8)
        
        return blocks - 128.0, (h, w)  # Center around 0
    
    def _reconstruct_image(self, blocks: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Reconstruct image from 8x8 blocks."""
        h, w = original_shape
        padded_h = ((h + 7) // 8) * 8
        padded_w = ((w + 7) // 8) * 8
        
        # Add 128 back and clip
        blocks = np.clip(blocks + 128.0, 0, 255)
        
        # Reshape blocks back to image
        blocks_reshaped = blocks.reshape(padded_h // 8, padded_w // 8, 8, 8)
        blocks_transposed = blocks_reshaped.transpose(0, 2, 1, 3)
        padded_image = blocks_transposed.reshape(padded_h, padded_w)
        
        return padded_image[:h, :w]
    
    def _sparse_encode(self, zigzag_data: np.ndarray) -> np.ndarray:
        """Encode zigzag data by truncating trailing zeros."""
        # For each block, find the last non-zero coefficient
        result_data = []
        for block_idx in range(zigzag_data.shape[0]):
            block = zigzag_data[block_idx]
            # Find last non-zero coefficient
            last_nonzero = 63
            while last_nonzero > 0 and block[last_nonzero] == 0:
                last_nonzero -= 1
            # Store only up to last non-zero coefficient + 1
            result_data.append(block[:last_nonzero + 1])
        return result_data
    
    def _sparse_decode(self, sparse_data: list, num_blocks: int) -> np.ndarray:
        """Decode sparse zigzag data back to full 64-coefficient blocks."""
        zigzag_data = np.zeros((num_blocks, 64), dtype=np.int16)
        for block_idx, block_data in enumerate(sparse_data):
            zigzag_data[block_idx, :len(block_data)] = block_data
        return zigzag_data
    
    def compress(self, image: np.ndarray, quality: int = 75, use_color: bool = True) -> Dict:
        """
        Compress image using reliable DCT with optional color optimization.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            quality: Compression quality 1-100 (higher = better quality)
            use_color: Whether to use YCbCr color space conversion
            
        Returns:
            Dictionary containing compressed data and metadata
        """
        # Quality scaling factor
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality
        scale = max(1, scale / 100.0)
        
        # Handle grayscale vs color
        if len(image.shape) == 2:
            # Grayscale compression
            quant_matrix = np.maximum(1, (self.luminance_quant * scale).astype(np.float32))
            
            blocks, orig_shape = self._extract_blocks(image)
            
            # Apply DCT
            dct_blocks = self._fast_dct_2d(blocks)
            
            # Quantize
            quantized = np.round(dct_blocks / quant_matrix).astype(np.int16)
            
            # Apply zigzag scan and sparse encoding
            zigzag_data = np.zeros((quantized.shape[0], 64), dtype=np.int16)
            for i in range(64):
                row, col = self.zigzag_indices[i] // 8, self.zigzag_indices[i] % 8
                zigzag_data[:, i] = quantized[:, row, col]
            
            # Apply sparse encoding (remove trailing zeros)
            sparse_data = self._sparse_encode(zigzag_data)
            
            return {
                'compressed_data': sparse_data,
                'quantization_matrix': quant_matrix,
                'original_shape': orig_shape,
                'num_channels': 1,
                'blocks_shape': quantized.shape[0],
                'use_color_transform': False
            }
        
        else:
            # Color image compression
            processed_image = image.copy().astype(np.float32)
            
            # Convert to YCbCr if requested
            if use_color:
                processed_image = self._rgb_to_ycbcr(processed_image)
            
            compressed_channels = []
            quantization_matrices = []
            
            for ch in range(image.shape[2]):
                channel = processed_image[:, :, ch]
                blocks, orig_shape = self._extract_blocks(channel)
                
                # Use different quantization for Y vs CbCr channels
                if use_color and ch == 0:  # Y channel (luminance)
                    quant_matrix = np.maximum(1, (self.luminance_quant * scale).astype(np.float32))
                elif use_color and ch > 0:  # CbCr channels (chrominance)
                    quant_matrix = np.maximum(1, (self.chrominance_quant * scale).astype(np.float32))
                else:  # RGB channels
                    quant_matrix = np.maximum(1, (self.luminance_quant * scale).astype(np.float32))
                
                quantization_matrices.append(quant_matrix)
                
                # Apply DCT
                dct_blocks = self._fast_dct_2d(blocks)
                
                # Quantize
                quantized = np.round(dct_blocks / quant_matrix).astype(np.int16)
                
                # Apply zigzag scan and sparse encoding
                zigzag_data = np.zeros((quantized.shape[0], 64), dtype=np.int16)
                for i in range(64):
                    row, col = self.zigzag_indices[i] // 8, self.zigzag_indices[i] % 8
                    zigzag_data[:, i] = quantized[:, row, col]
                
                # Apply sparse encoding (remove trailing zeros)
                sparse_data = self._sparse_encode(zigzag_data)
                compressed_channels.append(sparse_data)
            
            return {
                'compressed_channels': compressed_channels,
                'quantization_matrices': quantization_matrices,
                'original_shape': orig_shape,
                'num_channels': image.shape[2],
                'blocks_shape': quantized.shape[0],
                'use_color_transform': use_color
            }
    
    def decompress(self, compressed_data: Dict) -> np.ndarray:
        """Decompress DCT compressed image."""
        
        orig_shape = compressed_data['original_shape']
        num_blocks = compressed_data['blocks_shape']
        use_color_transform = compressed_data.get('use_color_transform', False)
        
        if compressed_data['num_channels'] == 1:
            # Grayscale decompression
            quant_matrix = compressed_data['quantization_matrix']
            sparse_data = compressed_data['compressed_data']
            
            # Decode sparse data
            zigzag_data = self._sparse_decode(sparse_data, num_blocks)
            
            # Inverse zigzag scan
            quantized = np.zeros((num_blocks, 8, 8), dtype=np.float32)
            for i in range(64):
                row, col = self.zigzag_indices[i] // 8, self.zigzag_indices[i] % 8
                quantized[:, row, col] = zigzag_data[:, i]
            
            # Dequantize
            dct_blocks = quantized * quant_matrix
            
            # Inverse DCT
            blocks = self._fast_idct_2d(dct_blocks)
            
            # Reconstruct image
            return self._reconstruct_image(blocks, orig_shape).astype(np.uint8)
        
        else:
            # Color decompression
            channels = []
            quant_matrices = compressed_data['quantization_matrices']
            
            for ch in range(compressed_data['num_channels']):
                sparse_data = compressed_data['compressed_channels'][ch]
                quant_matrix = quant_matrices[ch]
                
                # Decode sparse data
                zigzag_data = self._sparse_decode(sparse_data, num_blocks)
                
                # Inverse zigzag scan
                quantized = np.zeros((num_blocks, 8, 8), dtype=np.float32)
                for i in range(64):
                    row, col = self.zigzag_indices[i] // 8, self.zigzag_indices[i] % 8
                    quantized[:, row, col] = zigzag_data[:, i]
                
                # Dequantize
                dct_blocks = quantized * quant_matrix
                
                # Inverse DCT
                blocks = self._fast_idct_2d(dct_blocks)
                
                # Reconstruct channel
                channel = self._reconstruct_image(blocks, orig_shape)
                channels.append(channel)
            
            # Stack channels
            image = np.stack(channels, axis=2)
            
            # Convert back to RGB if YCbCr was used
            if use_color_transform:
                image = self._ycbcr_to_rgb(image)
            
            return np.clip(image, 0, 255).astype(np.uint8)
    
    def compress(self, image: np.ndarray, quality: int = 75, use_color: bool = True) -> Dict:
        """
        Compress image using simplified DCT.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            quality: Compression quality 1-100 (higher = better quality)
            
        Returns:
            Dictionary containing compressed data and metadata
        """
        # Quality scaling factor
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality
        scale = max(1, scale / 100.0)
        
        # Create quantization matrix
        quant_matrix = np.maximum(1, (self.luminance_quant * scale).astype(np.float32))
        
        # Handle grayscale vs color
        if len(image.shape) == 2:
            # Grayscale compression
            blocks, orig_shape = self._extract_blocks(image)
            
            # Apply DCT
            dct_blocks = self._fast_dct_2d(blocks)
            
            # Quantize
            quantized = np.round(dct_blocks / quant_matrix).astype(np.int16)
            
            # Apply zigzag scan
            zigzag_data = np.zeros((quantized.shape[0], 64), dtype=np.int16)
            for i in range(64):
                row, col = self.zigzag_indices[i] // 8, self.zigzag_indices[i] % 8
                zigzag_data[:, i] = quantized[:, row, col]
            
            return {
                'compressed_data': zigzag_data.flatten(),
                'quantization_matrix': quant_matrix,
                'original_shape': orig_shape,
                'num_channels': 1,
                'blocks_shape': quantized.shape[0]
            }
        
        else:
            # Color image compression
            compressed_channels = []
            
            for ch in range(image.shape[2]):
                channel = image[:, :, ch]
                blocks, orig_shape = self._extract_blocks(channel)
                
                # Apply DCT
                dct_blocks = self._fast_dct_2d(blocks)
                
                # Quantize
                quantized = np.round(dct_blocks / quant_matrix).astype(np.int16)
                
                # Apply zigzag scan
                zigzag_data = np.zeros((quantized.shape[0], 64), dtype=np.int16)
                for i in range(64):
                    row, col = self.zigzag_indices[i] // 8, self.zigzag_indices[i] % 8
                    zigzag_data[:, i] = quantized[:, row, col]
                
                compressed_channels.append(zigzag_data.flatten())
            
            return {
                'compressed_channels': compressed_channels,
                'quantization_matrix': quant_matrix,
                'original_shape': orig_shape,
                'num_channels': image.shape[2],
                'blocks_shape': quantized.shape[0]
            }
    
    def decompress(self, compressed_data: Dict) -> np.ndarray:
        """Decompress DCT compressed image."""
        
        quant_matrix = compressed_data['quantization_matrix']
        orig_shape = compressed_data['original_shape']
        num_blocks = compressed_data['blocks_shape']
        
        if compressed_data['num_channels'] == 1:
            # Grayscale decompression
            zigzag_flat = compressed_data['compressed_data']
            
            # Reshape zigzag data
            zigzag_data = zigzag_flat.reshape(num_blocks, 64)
            
            # Inverse zigzag scan
            quantized = np.zeros((num_blocks, 8, 8), dtype=np.float32)
            for i in range(64):
                row, col = self.zigzag_indices[i] // 8, self.zigzag_indices[i] % 8
                quantized[:, row, col] = zigzag_data[:, i]
            
            # Dequantize
            dct_blocks = quantized * quant_matrix
            
            # Inverse DCT
            blocks = self._fast_idct_2d(dct_blocks)
            
            # Reconstruct image
            return self._reconstruct_image(blocks, orig_shape).astype(np.uint8)
        
        else:
            # Color decompression
            channels = []
            
            for ch in range(compressed_data['num_channels']):
                zigzag_flat = compressed_data['compressed_channels'][ch]
                
                # Reshape zigzag data
                zigzag_data = zigzag_flat.reshape(num_blocks, 64)
                
                # Inverse zigzag scan
                quantized = np.zeros((num_blocks, 8, 8), dtype=np.float32)
                for i in range(64):
                    row, col = self.zigzag_indices[i] // 8, self.zigzag_indices[i] % 8
                    quantized[:, row, col] = zigzag_data[:, i]
                
                # Dequantize
                dct_blocks = quantized * quant_matrix
                
                # Inverse DCT
                blocks = self._fast_idct_2d(dct_blocks)
                
                # Reconstruct channel
                channel = self._reconstruct_image(blocks, orig_shape)
                channels.append(channel)
            
            # Stack channels
            image = np.stack(channels, axis=2)
            
            return np.clip(image, 0, 255).astype(np.uint8)

# Alias for backward compatibility
OptimizedDCTCompressor = SimplifiedDCTCompressor

def test_compressor():
    """Test the DCT compressor."""
    # Create test image
    test_image = np.random.randint(0, 256, (64, 64, 3)).astype(np.uint8)
    
    # Initialize compressor
    compressor = SimplifiedDCTCompressor()
    
    # Compress
    compressed = compressor.compress(test_image, quality=75)
    
    # Decompress
    reconstructed = compressor.decompress(compressed)
    
    print(f"Original shape: {test_image.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Calculate MSE and PSNR
    mse = np.mean((test_image.astype(float) - reconstructed.astype(float))**2)
    psnr = 20 * np.log10(255) - 10 * np.log10(mse) if mse > 0 else float('inf')
    
    print(f"MSE: {mse:.2f}")
    print(f"PSNR: {psnr:.2f} dB")
    
    return test_image, reconstructed, compressed

if __name__ == "__main__":
    test_compressor()