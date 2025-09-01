import numpy as np
from scipy.fft import dct, idct
import pickle
from typing import Tuple, Dict, Optional, Union, List
from collections import Counter
import zlib
import struct

class ImprovedDCTCompressor:
    """
    Enhanced DCT compressor with entropy coding and better small file handling.
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
    
    def _simple_entropy_encode(self, data: np.ndarray) -> bytes:
        """
        Simple entropy encoding using zlib compression on packed data.
        Much better than raw storage for sparse data.
        """
        # Convert to more compact representation
        # Use int8 for coefficients that fit, otherwise use run-length encoding
        
        # Flatten the data
        flat_data = data.flatten()
        
        # Simple approach: use zlib on the raw bytes after packing efficiently
        # Pack small values as int8, larger ones with a marker
        packed_data = bytearray()
        
        for val in flat_data:
            if -128 <= val <= 127:
                # Can fit in int8
                packed_data.extend(struct.pack('b', val))
            else:
                # Use marker byte (-128) followed by int16
                packed_data.extend(struct.pack('b', -128))
                packed_data.extend(struct.pack('<h', val))
        
        # Apply zlib compression
        return zlib.compress(packed_data)
    
    def _simple_entropy_decode(self, compressed_data: bytes, expected_size: int) -> np.ndarray:
        """Decode entropy-encoded data."""
        # Decompress with zlib
        packed_data = zlib.decompress(compressed_data)
        
        # Unpack the data
        values = []
        i = 0
        while i < len(packed_data):
            val = struct.unpack('b', packed_data[i:i+1])[0]
            i += 1
            
            if val == -128:
                # This is a marker for int16 value
                if i + 1 < len(packed_data):
                    val = struct.unpack('<h', packed_data[i:i+2])[0]
                    i += 2
                else:
                    val = 0  # Safety fallback
            
            values.append(val)
        
        # Pad or truncate to expected size
        while len(values) < expected_size:
            values.append(0)
        
        return np.array(values[:expected_size], dtype=np.int16)
    
    def _rgb_to_ycbcr(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB to YCbCr color space."""
        transform_matrix = np.array([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ], dtype=np.float32)
        
        h, w, c = rgb_image.shape
        rgb_flat = rgb_image.reshape(-1, 3).astype(np.float32)
        ycbcr_flat = np.dot(rgb_flat, transform_matrix.T)
        ycbcr_flat[:, 1:] += 128
        
        return ycbcr_flat.reshape(h, w, c)
    
    def _ycbcr_to_rgb(self, ycbcr_image: np.ndarray) -> np.ndarray:
        """Convert YCbCr back to RGB."""
        inv_transform_matrix = np.array([
            [1.0, 0.0, 1.402],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.772, 0.0]
        ], dtype=np.float32)
        
        h, w, c = ycbcr_image.shape
        ycbcr_flat = ycbcr_image.reshape(-1, 3).astype(np.float32)
        ycbcr_flat[:, 1:] -= 128
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
        
        padded_image = np.zeros((padded_h, padded_w), dtype=np.float32)
        padded_image[:h, :w] = image.astype(np.float32)
        
        blocks = padded_image.reshape(padded_h // 8, 8, padded_w // 8, 8)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, 8, 8)
        
        return blocks - 128.0, (h, w)
    
    def _reconstruct_image(self, blocks: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Reconstruct image from 8x8 blocks."""
        h, w = original_shape
        padded_h = ((h + 7) // 8) * 8
        padded_w = ((w + 7) // 8) * 8
        
        blocks = np.clip(blocks + 128.0, 0, 255)
        
        blocks_reshaped = blocks.reshape(padded_h // 8, padded_w // 8, 8, 8)
        blocks_transposed = blocks_reshaped.transpose(0, 2, 1, 3)
        padded_image = blocks_transposed.reshape(padded_h, padded_w)
        
        return padded_image[:h, :w]
    
    def should_compress(self, image: np.ndarray, quality: int = 75) -> bool:
        """
        Determine if compression would be beneficial.
        For very small images, compression might not help.
        """
        # Estimate original size (rough approximation)
        original_size = image.size * image.itemsize
        
        # Very rough estimate of compressed size
        # (This is a heuristic - in practice you might want to do a quick test compression)
        estimated_overhead = 1000  # bytes for metadata
        estimated_data_size = (image.size // 64) * 32  # rough DCT compression estimate
        estimated_compressed_size = estimated_overhead + estimated_data_size
        
        return estimated_compressed_size < original_size * 0.9  # Only compress if we save at least 10%
    
    def compress(self, image: np.ndarray, quality: int = 75, use_color: bool = True, 
                 force_compress: bool = False) -> Dict:
        """
        Compress image with better small file handling.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            quality: Compression quality 1-100
            use_color: Whether to use YCbCr color space conversion
            force_compress: Force compression even if it might not be beneficial
        """
        
        # Check if compression is worthwhile for small images
        if not force_compress and not self.should_compress(image, quality):
            # Return uncompressed data with a flag
            return {
                'uncompressed': True,
                'original_data': image,
                'compression_skipped': True
            }
        
        # Quality scaling
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality
        scale = max(1, scale / 100.0)
        
        if len(image.shape) == 2:
            # Grayscale compression
            quant_matrix = np.maximum(1, (self.luminance_quant * scale).astype(np.float32))
            blocks, orig_shape = self._extract_blocks(image)
            dct_blocks = self._fast_dct_2d(blocks)
            quantized = np.round(dct_blocks / quant_matrix).astype(np.int16)
            
            # Apply zigzag scan
            zigzag_data = np.zeros((quantized.shape[0], 64), dtype=np.int16)
            for i in range(64):
                row, col = self.zigzag_indices[i] // 8, self.zigzag_indices[i] % 8
                zigzag_data[:, i] = quantized[:, row, col]
            
            # Apply entropy encoding
            compressed_bytes = self._simple_entropy_encode(zigzag_data)
            
            return {
                'uncompressed': False,
                'compressed_data': compressed_bytes,
                'quantization_matrix': quant_matrix.astype(np.float16),  # Use float16 to save space
                'original_shape': orig_shape,
                'num_channels': 1,
                'blocks_shape': quantized.shape[0],
                'use_color_transform': False,
                'expected_coefficients': zigzag_data.size
            }
        
        else:
            # Color compression with entropy coding
            processed_image = image.copy().astype(np.float32)
            if use_color:
                processed_image = self._rgb_to_ycbcr(processed_image)
            
            compressed_channels = []
            quantization_matrices = []
            
            for ch in range(image.shape[2]):
                channel = processed_image[:, :, ch]
                blocks, orig_shape = self._extract_blocks(channel)
                
                if use_color and ch == 0:
                    quant_matrix = np.maximum(1, (self.luminance_quant * scale).astype(np.float32))
                elif use_color and ch > 0:
                    quant_matrix = np.maximum(1, (self.chrominance_quant * scale).astype(np.float32))
                else:
                    quant_matrix = np.maximum(1, (self.luminance_quant * scale).astype(np.float32))
                
                quantization_matrices.append(quant_matrix.astype(np.float16))
                
                dct_blocks = self._fast_dct_2d(blocks)
                quantized = np.round(dct_blocks / quant_matrix).astype(np.int16)
                
                # Apply zigzag scan
                zigzag_data = np.zeros((quantized.shape[0], 64), dtype=np.int16)
                for i in range(64):
                    row, col = self.zigzag_indices[i] // 8, self.zigzag_indices[i] % 8
                    zigzag_data[:, i] = quantized[:, row, col]
                
                # Apply entropy encoding
                compressed_bytes = self._simple_entropy_encode(zigzag_data)
                compressed_channels.append(compressed_bytes)
            
            return {
                'uncompressed': False,
                'compressed_channels': compressed_channels,
                'quantization_matrices': quantization_matrices,
                'original_shape': orig_shape,
                'num_channels': image.shape[2],
                'blocks_shape': quantized.shape[0],
                'use_color_transform': use_color,
                'expected_coefficients': zigzag_data.size
            }
    
    def decompress(self, compressed_data: Dict) -> np.ndarray:
        """Decompress with handling for uncompressed data."""
        
        # Check if data was stored uncompressed
        if compressed_data.get('uncompressed', False):
            return compressed_data['original_data']
        
        orig_shape = compressed_data['original_shape']
        num_blocks = compressed_data['blocks_shape']
        use_color_transform = compressed_data.get('use_color_transform', False)
        expected_coeffs = compressed_data['expected_coefficients']
        
        if compressed_data['num_channels'] == 1:
            # Grayscale decompression
            quant_matrix = compressed_data['quantization_matrix'].astype(np.float32)
            compressed_bytes = compressed_data['compressed_data']
            
            # Decode entropy-encoded data
            zigzag_flat = self._simple_entropy_decode(compressed_bytes, expected_coeffs)
            zigzag_data = zigzag_flat.reshape(num_blocks, 64)
            
            # Inverse zigzag scan
            quantized = np.zeros((num_blocks, 8, 8), dtype=np.float32)
            for i in range(64):
                row, col = self.zigzag_indices[i] // 8, self.zigzag_indices[i] % 8
                quantized[:, row, col] = zigzag_data[:, i]
            
            dct_blocks = quantized * quant_matrix
            blocks = self._fast_idct_2d(dct_blocks)
            
            return self._reconstruct_image(blocks, orig_shape).astype(np.uint8)
        
        else:
            # Color decompression
            channels = []
            quant_matrices = compressed_data['quantization_matrices']
            
            for ch in range(compressed_data['num_channels']):
                compressed_bytes = compressed_data['compressed_channels'][ch]
                quant_matrix = quant_matrices[ch].astype(np.float32)
                
                # Decode entropy-encoded data
                zigzag_flat = self._simple_entropy_decode(compressed_bytes, expected_coeffs)
                zigzag_data = zigzag_flat.reshape(num_blocks, 64)
                
                # Inverse zigzag scan
                quantized = np.zeros((num_blocks, 8, 8), dtype=np.float32)
                for i in range(64):
                    row, col = self.zigzag_indices[i] // 8, self.zigzag_indices[i] % 8
                    quantized[:, row, col] = zigzag_data[:, i]
                
                dct_blocks = quantized * quant_matrix
                blocks = self._fast_idct_2d(dct_blocks)
                channel = self._reconstruct_image(blocks, orig_shape)
                channels.append(channel)
            
            image = np.stack(channels, axis=2)
            
            if use_color_transform:
                image = self._ycbcr_to_rgb(image)
            
            return np.clip(image, 0, 255).astype(np.uint8)

# def test_improved_compressor():
#     """Test the improved compressor with different image sizes."""
#     # Test with small image
#     small_image = np.random.randint(0, 256, (32, 32, 3)).astype(np.uint8)
    
#     # Test with larger image
#     large_image = np.random.randint(0, 256, (256, 256, 3)).astype(np.uint8)
    
#     compressor = ImprovedDCTCompressor()
    
#     for name, test_image in [("Small", small_image), ("Large", large_image)]:
#         print(f"\n{name} Image Test:")
#         print(f"Original shape: {test_image.shape}")
#         print(f"Original size estimate: {test_image.nbytes} bytes")
        
#         # Compress
#         compressed = compressor.compress(test_image, quality=75)
        
#         if compressed.get('uncompressed', False):
#             print("Compression skipped (not beneficial)")
#             compressed_size = test_image.nbytes
#         else:
#             # Estimate compressed size (rough)
#             if 'compressed_data' in compressed:
#                 compressed_size = len(compressed['compressed_data']) + 512  # + metadata
#             else:
#                 compressed_size = sum(len(data) for data in compressed['compressed_channels']) + 1024
            
#             print(f"Estimated compressed size: {compressed_size} bytes")
        
#         # Decompress
#         reconstructed = compressor.decompress(compressed)
#         print(f"Reconstructed shape: {reconstructed.shape}")
        
#         # Calculate quality metrics
#         mse = np.mean((test_image.astype(float) - reconstructed.astype(float))**2)
#         psnr = 20 * np.log10(255) - 10 * np.log10(mse) if mse > 0 else float('inf')
#         print(f"MSE: {mse:.2f}")
#         print(f"PSNR: {psnr:.2f} dB")

# if __name__ == "__main__":
#     test_improved_compressor()