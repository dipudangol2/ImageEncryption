"""
Unified compression module that provides a consistent API across the system.
Uses the ImprovedDCTCompressor as the standard implementation.
"""

from improved_compressor import ImprovedDCTCompressor
from typing import Dict, Tuple
import numpy as np

class UnifiedCompressor:
    """
    Unified compression interface that wraps ImprovedDCTCompressor
    and provides backward compatibility with existing code.
    """
    
    def __init__(self):
        self.compressor = ImprovedDCTCompressor()
    
    def compress(self, image: np.ndarray, quality: int = 75, use_color: bool = True) -> Dict:
        """
        Compress image using the improved DCT compressor.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            quality: Compression quality 1-100
            use_color: Whether to use YCbCr color space conversion
            
        Returns:
            Dictionary containing compressed data and metadata
        """
        return self.compressor.compress(image, quality, use_color, force_compress=True)
    
    def decompress(self, compressed_data: Dict) -> np.ndarray:
        """
        Decompress image data.
        
        Args:
            compressed_data: Dictionary from compress() method
            
        Returns:
            Reconstructed image as numpy array
        """
        return self.compressor.decompress(compressed_data)
    
    def get_compression_stats(self, original_image: np.ndarray, 
                            compressed_data: Dict) -> Dict:
        """
        Calculate compression statistics.
        
        Args:
            original_image: Original image array
            compressed_data: Compressed data dictionary
            
        Returns:
            Dictionary with compression statistics
        """
        original_size = original_image.nbytes
        
        # Calculate compressed size
        if compressed_data.get('uncompressed', False):
            compressed_size = original_size
        else:
            compressed_size = 0
            if 'compressed_data' in compressed_data:
                compressed_size += len(compressed_data['compressed_data'])
            elif 'compressed_channels' in compressed_data:
                for channel_data in compressed_data['compressed_channels']:
                    compressed_size += len(channel_data)
            
            # Add metadata overhead estimate
            compressed_size += 512
        
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio
        }