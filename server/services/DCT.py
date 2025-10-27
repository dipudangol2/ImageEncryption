"""
Self-contained Manual DCT Compressor with Quantization.

Features:
- Manual DCT implementation (no external imports)
- JPEG-style quantization matrices for real compression
- Compact storage format
- Fixed unit conversion for accurate file size reporting
"""

import numpy as np
import time
from typing import Dict, Tuple


class DCTCompressor:
    """
    Complete self-contained DCT image compressor with quantization.

    Implements manual DCT transformation and JPEG-style quantization
    for actual file size reduction, not just quality loss.
    """

    def __init__(self):
        """Initialize DCT matrices and quantization tables."""
        self.dct_matrix = self._create_dct_matrix()
        self.idct_matrix = self.dct_matrix.T  # IDCT is transpose of DCT

        # JPEG quantization matrices
        self.luminance_quant = np.array(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ],
            dtype=np.float32,
        )

        self.chrominance_quant = np.array(
            [
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
            ],
            dtype=np.float32,
        )

    def _create_dct_matrix(self) -> np.ndarray:
        """Create 8x8 DCT transformation matrix."""
        N = 8
        dct_matrix = np.zeros((N, N), dtype=np.float32)

        for u in range(N):
            for x in range(N):
                if u == 0:
                    dct_matrix[u, x] = np.sqrt(1.0 / N)
                else:
                    dct_matrix[u, x] = np.sqrt(2.0 / N) * np.cos(
                        (2 * x + 1) * u * np.pi / (2 * N)
                    )

        return dct_matrix

    def _dct2d(self, block: np.ndarray) -> np.ndarray:
        """Apply 2D DCT to 8x8 block."""
        return self.dct_matrix @ block @ self.dct_matrix.T

    def _idct2d(self, dct_block: np.ndarray) -> np.ndarray:
        """Apply 2D inverse DCT to 8x8 block."""
        return self.idct_matrix @ dct_block @ self.idct_matrix.T

    def _get_quantization_matrix(
        self, quality: int, is_luminance: bool = True
    ) -> np.ndarray:
        """Generate quantization matrix based on quality (1-100)."""
        # Quality scaling (JPEG standard)
        if quality < 50:
            scale = 5000.0 / quality
        else:
            scale = 200.0 - 2.0 * quality

        scale = max(1.0, scale / 100.0)

        # Choose base quantization matrix
        base_matrix = self.luminance_quant if is_luminance else self.chrominance_quant

        # Apply scaling and ensure minimum value of 1
        quant_matrix = np.maximum(1, (base_matrix * scale).astype(np.int32))

        return quant_matrix.astype(np.float32)

    def _pad_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Pad image to be divisible by 8x8 blocks."""
        if len(image.shape) == 2:
            h, w = image.shape
            channels = 1
        else:
            h, w, channels = image.shape

        # Calculate padding needed
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        if len(image.shape) == 2:
            padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")
        else:
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

        return padded, (h, w)

    def _compress_channel(
        self, channel: np.ndarray, quality: int, is_luminance: bool = True
    ) -> Tuple[np.ndarray, bytes]:
        """Compress a single channel using DCT + quantization."""
        # Pad channel and shift to [-128, 127]
        padded_channel, original_shape = self._pad_image(channel)
        channel_float = padded_channel.astype(np.float32) - 128.0

        h, w = channel_float.shape

        # Get quantization matrix
        quant_matrix = self._get_quantization_matrix(quality, is_luminance)

        # Store quantized coefficients compactly (only non-zero ones)
        compressed_blocks = []

        # Process in 8x8 blocks
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                # Extract block
                block = channel_float[i : i + 8, j : j + 8]

                # Apply DCT
                dct_coeffs = self._dct2d(block)

                # Quantize
                quantized = np.round(dct_coeffs / quant_matrix).astype(np.int16)

                # Store only non-zero coefficients with their positions
                flat_quantized = quantized.flatten()
                non_zero_indices = np.nonzero(flat_quantized)[0]
                non_zero_values = flat_quantized[non_zero_indices]

                # Store block info: (num_coeffs, indices, values)
                compressed_blocks.append(
                    {
                        "indices": non_zero_indices.astype(
                            np.uint8
                        ),  # Position in 64-element block
                        "values": non_zero_values.astype(np.int16),  # Quantized values
                    }
                )

        # Serialize efficiently
        import struct

        coeffs_data = bytearray()

        # Store number of blocks first
        coeffs_data.extend(struct.pack("<I", len(compressed_blocks)))

        for block in compressed_blocks:
            indices = block["indices"]
            values = block["values"]
            num_coeffs = len(indices)

            # Store: num_coeffs (1 byte), indices (num_coeffs bytes), values (num_coeffs * 2 bytes)
            coeffs_data.extend(struct.pack("<B", num_coeffs))  # Max 64 coeffs per block
            coeffs_data.extend(indices.tobytes())
            coeffs_data.extend(values.tobytes())

        coeffs_bytes = bytes(coeffs_data)

        # Reconstruct for preview using compressed data
        reconstructed = self._reconstruct_from_compressed_blocks(
            compressed_blocks, (h, w), quant_matrix
        )

        # Remove padding
        reconstructed = reconstructed[: original_shape[0], : original_shape[1]]

        return reconstructed.astype(np.uint8), coeffs_bytes

    def _reconstruct_from_compressed_blocks(
        self,
        compressed_blocks: list,
        padded_shape: Tuple[int, int],
        quant_matrix: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct image from compressed block data."""
        h, w = padded_shape
        reconstructed = np.zeros((h, w), dtype=np.float32)

        block_idx = 0
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                if block_idx < len(compressed_blocks):
                    block_data = compressed_blocks[block_idx]

                    # Reconstruct 8x8 block from sparse coefficients
                    quantized_block = np.zeros(64, dtype=np.float32)
                    indices = block_data["indices"]
                    values = block_data["values"].astype(np.float32)
                    quantized_block[indices] = values
                    quantized_block = quantized_block.reshape(8, 8)

                    # Dequantize
                    dequantized = quantized_block * quant_matrix

                    # Apply IDCT
                    reconstructed_block = self._idct2d(dequantized)

                    # Store in image
                    reconstructed[i : i + 8, j : j + 8] = reconstructed_block

                block_idx += 1

        # Shift back to [0, 255]
        reconstructed = np.clip(reconstructed + 128.0, 0, 255)
        return reconstructed

    def quality_to_compression_ratio(self, quality: int) -> float:
        """Convert API quality (1-100) to compression ratio for compatibility."""
        return quality / 100.0

    def compress(
        self, image: np.ndarray, quality: int = 60, use_color: bool = True
    ) -> Dict:
        start_time = time.time()
        quality = max(1, min(100, quality))
        if len(image.shape) == 2:
            reconstructed, coeffs_bytes = self._compress_channel(image, quality, True)
            compressed_data = {
                "quantized_coefficients": coeffs_bytes,
                "quantization_matrix": self._get_quantization_matrix(
                    quality, True
                ).astype(np.float16),
                "original_shape": image.shape,
                "quality": quality,
                "num_channels": 1,
                "use_manual_dct": True,
                "use_quantization": True,
                "processing_time": time.time() - start_time,
            }
        else:
            h, w, c = image.shape
            reconstructed_channels = []
            all_coeffs_bytes = []
            for ch in range(c):
                channel = image[:, :, ch]
                reconstructed_ch, coeffs_bytes = self._compress_channel(
                    channel, quality, True
                )
                reconstructed_channels.append(reconstructed_ch)
                all_coeffs_bytes.append(coeffs_bytes)
            compressed_data = {
                "quantized_coefficients": all_coeffs_bytes,
                "quantization_matrix": self._get_quantization_matrix(
                    quality, True
                ).astype(np.float16),
                "original_shape": image.shape,
                "quality": quality,
                "num_channels": c,
                "use_manual_dct": True,
                "use_quantization": True,
                "processing_time": time.time() - start_time,
            }
        return compressed_data

    def decompress(self, compressed_data: Dict) -> np.ndarray:
        original_shape = compressed_data["original_shape"]
        quality = compressed_data["quality"]
        num_channels = compressed_data["num_channels"]
        coeffs = compressed_data["quantized_coefficients"]

        if num_channels == 1:
            # Grayscale reconstruction
            return self._decompress_channel(coeffs, original_shape, quality, True)
        else:
            # Color reconstruction
            h, w, c = original_shape
            channels = []

            for ch in range(c):
                channel_coeffs = coeffs[ch]
                reconstructed_ch = self._decompress_channel(
                    channel_coeffs, (h, w), quality, True
                )
                channels.append(reconstructed_ch)

            return np.stack(channels, axis=2)

    def get_compressed_preview(self, compressed_data: Dict) -> np.ndarray:
       # This is identical to decompress() but clearly labeled for testing
        return self.decompress(compressed_data)

    def _decompress_channel(
        self,
        coeffs_bytes: bytes,
        original_shape: Tuple[int, int],
        quality: int,
        is_luminance: bool = True,
    ) -> np.ndarray:
        """Decompress a single channel from compressed coefficients."""
        h, w = original_shape

        # Pad to 8x8 blocks
        padded_h = ((h + 7) // 8) * 8
        padded_w = ((w + 7) // 8) * 8

        # Get quantization matrix
        quant_matrix = self._get_quantization_matrix(quality, is_luminance)

        # Parse compressed data
        import struct

        compressed_blocks = []
        offset = 0

        # Read number of blocks
        num_blocks = struct.unpack("<I", coeffs_bytes[offset : offset + 4])[0]
        offset += 4

        for _ in range(num_blocks):
            # Read number of coefficients
            num_coeffs = struct.unpack("<B", coeffs_bytes[offset : offset + 1])[0]
            offset += 1

            # Read indices
            indices = np.frombuffer(
                coeffs_bytes[offset : offset + num_coeffs], dtype=np.uint8
            )
            offset += num_coeffs

            # Read values
            values = np.frombuffer(
                coeffs_bytes[offset : offset + num_coeffs * 2], dtype=np.int16
            )
            offset += num_coeffs * 2

            compressed_blocks.append({"indices": indices, "values": values})

        # Reconstruct image
        reconstructed = self._reconstruct_from_compressed_blocks(
            compressed_blocks, (padded_h, padded_w), quant_matrix
        )

        # Remove padding
        reconstructed = reconstructed[:h, :w]
        return reconstructed.astype(np.uint8)

    def get_compression_stats(
        self, original_image: np.ndarray, compressed_data: Dict
    ) -> Dict:
        # Original image size in bytes
        original_size_bytes = original_image.nbytes

        # Calculate actual compressed data size (only coefficients + metadata)
        compressed_size_bytes = 0

        # Size of quantized coefficients (the actual compressed data)
        if "quantized_coefficients" in compressed_data:
            coeffs = compressed_data["quantized_coefficients"]
            if isinstance(coeffs, list):  # Multi-channel
                for coeff_bytes in coeffs:
                    compressed_size_bytes += len(coeff_bytes)
            else:  # Single channel
                compressed_size_bytes += len(coeffs)

        # Add quantization matrix size
        if "quantization_matrix" in compressed_data:
            compressed_size_bytes += compressed_data["quantization_matrix"].nbytes

        # Add metadata overhead
        metadata_size = 512  # Estimated metadata size in bytes
        total_compressed_size = compressed_size_bytes + metadata_size

        # Calculate ratios
        compression_ratio = (
            original_size_bytes / total_compressed_size
            if total_compressed_size > 0
            else 1.0
        )
        space_saved_percent = (
            (1 - total_compressed_size / original_size_bytes) * 100
            if original_size_bytes > 0
            else 0
        )

        # FIXED: Convert bytes to appropriate units for display
        def format_size(size_bytes):
            if size_bytes < 1024:
                return size_bytes, "bytes"
            elif size_bytes < 1024 * 1024:
                return round(size_bytes / 1024, 1), "KB"
            else:
                return round(size_bytes / (1024 * 1024), 1), "MB"

        original_size_display, original_unit = format_size(original_size_bytes)
        compressed_size_display, compressed_unit = format_size(total_compressed_size)

        return {
            # Raw bytes for calculations
            "original_size": original_size_bytes,
            "compressed_size": total_compressed_size,
            "compression_ratio": compression_ratio,
            "space_saved_percent": space_saved_percent,
            # Formatted for display (FIXES the unit bug)
            "original_size_display": f"{original_size_display} {original_unit}",
            "compressed_size_display": f"{compressed_size_display} {compressed_unit}",
            # Additional info
            "actual_coefficients_size": compressed_size_bytes,
            "quality": compressed_data.get("quality", 75),
        }

    def calculate_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        if len(original.shape) == 2:
            # Grayscale
            mse = np.mean(
                (original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2
            )
        else:
            # Color - calculate MSE across all channels
            mse = np.mean(
                (original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2
            )

        if mse == 0:
            return float("inf")
        return 20 * np.log10(255.0 / np.sqrt(mse))

