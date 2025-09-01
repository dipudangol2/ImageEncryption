import numpy as np
import math
from collections import defaultdict, Counter
import heapq
import pickle
from typing import Dict, List, Tuple, Optional
import struct

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

class ManualDCTCompressor:
    """
    Manual DCT implementation with Huffman coding, compatible with existing main script.
    """
    
    def __init__(self):
        # Pre-compute DCT basis functions for 8x8 blocks
        self.dct_matrix = self._compute_dct_matrix(8)
        self.idct_matrix = self.dct_matrix.T
        
        # JPEG quantization tables
        self.luma_quant = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)
        
        self.chroma_quant = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=np.float32)
        
        # Zigzag pattern
        self.zigzag_order = [
            (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
            (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
            (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
            (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
            (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
            (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
            (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
            (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
        ]
    
    def _compute_dct_matrix(self, N: int) -> np.ndarray:
        """Compute DCT transformation matrix manually."""
        dct_matrix = np.zeros((N, N), dtype=np.float32)
        
        for k in range(N):
            for n in range(N):
                if k == 0:
                    dct_matrix[k, n] = math.sqrt(1.0 / N)
                else:
                    dct_matrix[k, n] = math.sqrt(2.0 / N) * math.cos(
                        math.pi * k * (2 * n + 1) / (2 * N)
                    )
        
        return dct_matrix
    
    def _manual_dct_2d(self, block: np.ndarray) -> np.ndarray:
        """Manual 2D DCT using matrix multiplication."""
        return np.dot(self.dct_matrix, np.dot(block, self.dct_matrix.T))
    
    def _manual_idct_2d(self, dct_block: np.ndarray) -> np.ndarray:
        """Manual 2D inverse DCT."""
        return np.dot(self.idct_matrix, np.dot(dct_block, self.dct_matrix))
    
    def _rgb_to_ycbcr(self, rgb: np.ndarray) -> np.ndarray:
        """Manual RGB to YCbCr conversion."""
        rgb_float = rgb.astype(np.float32)
        
        Y = 0.299 * rgb_float[:, :, 0] + 0.587 * rgb_float[:, :, 1] + 0.114 * rgb_float[:, :, 2]
        Cb = -0.168736 * rgb_float[:, :, 0] - 0.331264 * rgb_float[:, :, 1] + 0.5 * rgb_float[:, :, 2] + 128
        Cr = 0.5 * rgb_float[:, :, 0] - 0.418688 * rgb_float[:, :, 1] - 0.081312 * rgb_float[:, :, 2] + 128
        
        return np.stack([Y, Cb, Cr], axis=2)
    
    def _ycbcr_to_rgb(self, ycbcr: np.ndarray) -> np.ndarray:
        """Manual YCbCr to RGB conversion."""
        Y = ycbcr[:, :, 0]
        Cb = ycbcr[:, :, 1] - 128
        Cr = ycbcr[:, :, 2] - 128
        
        R = Y + 1.402 * Cr
        G = Y - 0.344136 * Cb - 0.714136 * Cr
        B = Y + 1.772 * Cb
        
        rgb = np.stack([R, G, B], axis=2)
        return np.clip(rgb, 0, 255).astype(np.uint8)
    
    def _build_huffman_tree(self, frequencies: Dict[int, int]) -> HuffmanNode:
        """Build Huffman tree."""
        if len(frequencies) <= 1:
            symbol = next(iter(frequencies.keys()))
            return HuffmanNode(char=symbol, freq=frequencies[symbol])
        
        heap = []
        for char, freq in frequencies.items():
            heapq.heappush(heap, HuffmanNode(char=char, freq=freq))
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)
        
        return heap[0]
    
    def _generate_huffman_codes(self, root: HuffmanNode) -> Dict[int, str]:
        """Generate Huffman codes."""
        if root.char is not None:
            return {root.char: '0'}
        
        codes = {}
        def traverse(node, code=''):
            if node.char is not None:
                codes[node.char] = code
            else:
                if node.left:
                    traverse(node.left, code + '0')
                if node.right:
                    traverse(node.right, code + '1')
        
        traverse(root)
        return codes
    
    def _huffman_encode(self, data: List[int]) -> Tuple[bytes, HuffmanNode]:
        """Huffman encode data."""
        frequencies = Counter(data)
        root = self._build_huffman_tree(frequencies)
        codes = self._generate_huffman_codes(root)
        
        # Encode
        encoded_bits = ''.join(codes[symbol] for symbol in data)
        
        # Pad to byte boundary
        padding = 8 - (len(encoded_bits) % 8)
        if padding != 8:
            encoded_bits += '0' * padding
        
        # Convert to bytes
        encoded_bytes = bytearray()
        for i in range(0, len(encoded_bits), 8):
            byte = encoded_bits[i:i+8]
            encoded_bytes.append(int(byte, 2))
        
        return bytes(encoded_bytes), root
    
    def _huffman_decode(self, encoded_bytes: bytes, root: HuffmanNode, 
                       original_length: int) -> List[int]:
        """Huffman decode data."""
        if root.char is not None:
            return [root.char] * original_length
        
        bit_string = ''.join(format(byte, '08b') for byte in encoded_bytes)
        decoded = []
        current = root
        
        for bit in bit_string:
            if len(decoded) >= original_length:
                break
                
            current = current.left if bit == '0' else current.right
            
            if current.char is not None:
                decoded.append(current.char)
                current = root
        
        return decoded
    
    def _extract_blocks(self, channel: np.ndarray) -> Tuple[List[np.ndarray], Tuple[int, int]]:
        """Extract 8x8 blocks."""
        h, w = channel.shape
        pad_h = ((h + 7) // 8) * 8
        pad_w = ((w + 7) // 8) * 8
        
        padded = np.zeros((pad_h, pad_w), dtype=np.float32)
        padded[:h, :w] = channel.astype(np.float32) - 128
        
        blocks = []
        for i in range(0, pad_h, 8):
            for j in range(0, pad_w, 8):
                blocks.append(padded[i:i+8, j:j+8])
        
        return blocks, (h, w)
    
    def _reconstruct_from_blocks(self, blocks: List[np.ndarray], 
                                original_shape: Tuple[int, int]) -> np.ndarray:
        """Reconstruct image from blocks."""
        h, w = original_shape
        pad_h = ((h + 7) // 8) * 8
        pad_w = ((w + 7) // 8) * 8
        
        reconstructed = np.zeros((pad_h, pad_w), dtype=np.float32)
        
        block_idx = 0
        for i in range(0, pad_h, 8):
            for j in range(0, pad_w, 8):
                reconstructed[i:i+8, j:j+8] = blocks[block_idx] + 128
                block_idx += 1
        
        return np.clip(reconstructed[:h, :w], 0, 255).astype(np.uint8)
    
    def _zigzag_scan(self, block: np.ndarray) -> List[int]:
        """Apply zigzag scanning."""
        return [int(block[i, j]) for i, j in self.zigzag_order]
    
    def _inverse_zigzag(self, coeffs: List[int]) -> np.ndarray:
        """Reverse zigzag scanning."""
        block = np.zeros((8, 8), dtype=np.int16)
        for idx, (i, j) in enumerate(self.zigzag_order):
            if idx < len(coeffs):
                block[i, j] = coeffs[idx]
        return block
    
    def compress(self, image: np.ndarray, quality: int = 75, use_color: bool = True) -> Dict:
        """
        Compress image - COMPATIBLE with main script signature.
        """
        # Quality scaling
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality
        scale = max(1, scale / 100.0)
        
        # Handle grayscale
        if len(image.shape) == 2:
            # Convert grayscale to 3-channel for consistent processing
            image = np.stack([image, image, image], axis=2)
            use_color = False
        
        # Color space conversion
        if use_color:
            processed_image = self._rgb_to_ycbcr(image)
        else:
            processed_image = image.astype(np.float32)
        
        compressed_channels = []
        huffman_trees = []
        original_lengths = []
        quant_matrices = []
        
        for channel_idx in range(3):
            channel = processed_image[:, :, channel_idx]
            
            # Choose quantization table
            if use_color and channel_idx == 0:  # Y channel
                quant_table = np.maximum(1, self.luma_quant * scale)
            elif use_color and channel_idx > 0:  # Cb, Cr channels
                quant_table = np.maximum(1, self.chroma_quant * scale)
            else:  # RGB or repeated grayscale channels
                quant_table = np.maximum(1, self.luma_quant * scale)
            
            quant_matrices.append(quant_table)
            
            # Extract blocks
            blocks, orig_shape = self._extract_blocks(channel)
            
            # Process blocks
            all_coefficients = []
            
            for block in blocks:
                # Manual DCT
                dct_block = self._manual_dct_2d(block)
                
                # Quantize
                quantized = np.round(dct_block / quant_table).astype(np.int16)
                
                # Zigzag scan
                coeffs = self._zigzag_scan(quantized)
                all_coefficients.extend(coeffs)
            
            # Huffman encode
            try:
                encoded_data, huffman_tree = self._huffman_encode(all_coefficients)
                compressed_channels.append(encoded_data)
                huffman_trees.append(huffman_tree)
                original_lengths.append(len(all_coefficients))
            except Exception as e:
                # Fallback to simple encoding if Huffman fails
                print(f"Huffman encoding failed for channel {channel_idx}, using simple encoding: {e}")
                simple_encoded = struct.pack(f'{len(all_coefficients)}h', *all_coefficients)
                compressed_channels.append(simple_encoded)
                huffman_trees.append(None)
                original_lengths.append(len(all_coefficients))
        
        return {
            'compressed_channels': compressed_channels,
            'huffman_trees': huffman_trees,
            'original_lengths': original_lengths,
            'quant_matrices': quant_matrices,
            'original_shape': orig_shape,
            'num_channels': 3,
            'use_color_transform': use_color,
            'blocks_per_channel': len(blocks)
        }
    
    def decompress(self, compressed_data: Dict) -> np.ndarray:
        """Decompress the image - COMPATIBLE with main script."""
        channels = compressed_data['compressed_channels']
        huffman_trees = compressed_data['huffman_trees']
        original_lengths = compressed_data['original_lengths']
        quant_matrices = compressed_data['quant_matrices']
        orig_shape = compressed_data['original_shape']
        use_color = compressed_data['use_color_transform']
        blocks_per_channel = compressed_data['blocks_per_channel']
        
        reconstructed_channels = []
        
        for channel_idx in range(3):
            encoded_data = channels[channel_idx]
            huffman_tree = huffman_trees[channel_idx]
            orig_length = original_lengths[channel_idx]
            quant_table = quant_matrices[channel_idx]
            
            # Decode coefficients
            if huffman_tree is not None:
                # Huffman decode
                try:
                    coefficients = self._huffman_decode(encoded_data, huffman_tree, orig_length)
                except Exception as e:
                    print(f"Huffman decoding failed for channel {channel_idx}: {e}")
                    # Fallback
                    coefficients = [0] * orig_length
            else:
                # Simple decode
                coefficients = list(struct.unpack(f'{orig_length}h', encoded_data))
            
            # Reconstruct blocks
            reconstructed_blocks = []
            
            for block_idx in range(blocks_per_channel):
                start_idx = block_idx * 64
                block_coeffs = coefficients[start_idx:start_idx + 64]
                
                # Pad if necessary
                while len(block_coeffs) < 64:
                    block_coeffs.append(0)
                
                # Inverse zigzag
                quantized_block = self._inverse_zigzag(block_coeffs)
                
                # Dequantize
                dct_block = quantized_block.astype(np.float32) * quant_table
                
                # Manual inverse DCT
                spatial_block = self._manual_idct_2d(dct_block)
                reconstructed_blocks.append(spatial_block)
            
            # Reconstruct channel
            channel_reconstructed = self._reconstruct_from_blocks(
                reconstructed_blocks, orig_shape
            )
            reconstructed_channels.append(channel_reconstructed)
        
        # Stack channels
        if len(reconstructed_channels) == 1:
            # Grayscale
            return reconstructed_channels[0]
        else:
            # Color
            reconstructed = np.stack(reconstructed_channels, axis=2)
            
            if use_color:
                # Convert back to RGB
                reconstructed = self._ycbcr_to_rgb(reconstructed)
            
            return reconstructed
    
    def get_compression_ratio(self, original_image: np.ndarray, 
                            compressed_data: Dict) -> float:
        """Calculate compression ratio - REQUIRED by main script."""
        original_size = original_image.nbytes
        
        # Calculate compressed size
        compressed_size = 0
        for channel_data in compressed_data['compressed_channels']:
            compressed_size += len(channel_data)
        
        # Add size of quantization matrices
        for quant_matrix in compressed_data['quant_matrices']:
            compressed_size += quant_matrix.nbytes
        
        # Add rough overhead for other metadata
        compressed_size += 512
        
        return original_size / compressed_size if compressed_size > 0 else 0.0

# For compatibility with your original imports
OptimizedDCTCompressor = ManualDCTCompressor

def test_with_actual_image():
    """Test with a real image pattern."""
    # Create a more realistic test image
    size = 64
    test_image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create gradient patterns
    for i in range(size):
        for j in range(size):
            test_image[i, j, 0] = int(255 * (i / size))  # Red gradient
            test_image[i, j, 1] = int(255 * (j / size))  # Green gradient
            test_image[i, j, 2] = int(255 * ((i + j) / (2 * size)))  # Blue gradient
    
    compressor = ManualDCTCompressor()
    
    print("Testing Manual DCT Compressor")
    print(f"Original image shape: {test_image.shape}")
    print(f"Original size: {test_image.nbytes} bytes")
    
    # Compress
    compressed = compressor.compress(test_image, quality=75, use_color=True)
    
    # Get compression ratio
    ratio = compressor.get_compression_ratio(test_image, compressed)
    print(f"Compression ratio: {ratio:.2f}:1")
    
    # Decompress
    reconstructed = compressor.decompress(compressed)
    
    # Quality metrics
    mse = np.mean((test_image.astype(float) - reconstructed.astype(float))**2)
    psnr = 20 * np.log10(255) - 10 * np.log10(mse) if mse > 0 else float('inf')
    
    print(f"MSE: {mse:.2f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Shapes match: {test_image.shape == reconstructed.shape}")

if __name__ == "__main__":
    test_with_actual_image()