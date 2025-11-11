"""
Performance Analysis Module for Image Encryption System.

This module provides comprehensive performance metrics including:
- PSNR (Peak Signal-to-Noise Ratio)
- NPCR (Number of Pixels Change Rate) 
- UACI (Unified Average Changing Intensity)
- Histogram Analysis
- Execution Time Tracking
"""

import numpy as np
import time
from typing import Dict, Tuple, Any
from contextlib import contextmanager


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for image encryption and compression systems.
    """
    
    def __init__(self):
        """Initialize the performance analyzer."""
        self.timing_data = {}
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """
        Context manager for timing operations.
        
        Usage:
            with analyzer.time_operation("compression"):
                # do compression work
                pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            self.timing_data[operation_name] = end_time - start_time
    
    def calculate_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        PSNR measures the quality of reconstruction compared to original.
        Higher PSNR indicates better quality (less distortion).
        
        Args:
            original: Original image
            reconstructed: Reconstructed/compressed image
            
        Returns:
            PSNR value in dB
        """
        if len(original.shape) == 2:
            # Grayscale
            mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
        else:
            # Color - calculate MSE across all channels
            mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
        
        if mse == 0:
            return float('inf')  # Perfect reconstruction
        
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def calculate_npcr(self, original: np.ndarray, encrypted_visualization: np.ndarray) -> float:
        """
        Calculate Number of Pixels Change Rate (NPCR).
        
        NPCR measures the percentage of different pixels between two images.
        Higher NPCR indicates better encryption (more pixels changed).
        Ideal NPCR for good encryption should be close to 99.6%.
        
        Args:
            original: Original image
            encrypted_visualization: Encrypted image visualization
            
        Returns:
            NPCR percentage (0-100)
        """
        # Ensure images have same dimensions for comparison
        encrypted_resized = self._resize_to_match(encrypted_visualization, original.shape)
        
        # Calculate different pixels
        different_pixels = np.sum(original != encrypted_resized)
        total_pixels = original.size
        
        npcr = (different_pixels / total_pixels) * 100
        return npcr
    
    def calculate_uaci(self, original: np.ndarray, encrypted_visualization: np.ndarray) -> float:
        """
        Calculate Unified Average Changing Intensity (UACI).
        
        UACI measures the average intensity of differences between corresponding pixels.
        Higher UACI indicates better encryption (larger intensity differences).
        Ideal UACI for good encryption should be close to 33.46%.
        
        Args:
            original: Original image
            encrypted_visualization: Encrypted image visualization
            
        Returns:
            UACI percentage (0-100)
        """
        # Ensure images have same dimensions for comparison
        encrypted_resized = self._resize_to_match(encrypted_visualization, original.shape)
        
        # Calculate UACI
        original_float = original.astype(np.float32)
        encrypted_float = encrypted_resized.astype(np.float32)
        
        differences = np.abs(original_float - encrypted_float)
        uaci = np.mean(differences) / 255.0 * 100
        
        return uaci
    
    def analyze_histogram(self, image: np.ndarray) -> Dict:
        """
        Perform comprehensive histogram analysis.
        
        Provides detailed statistical information about pixel intensity distribution.
        
        Args:
            image: Input image for analysis
            
        Returns:
            Dictionary containing histogram statistics
        """
        if len(image.shape) == 2:
            return self._analyze_grayscale_histogram(image)
        else:
            return self._analyze_color_histogram(image)
    
    def _analyze_grayscale_histogram(self, image: np.ndarray) -> Dict:
        """Analyze histogram for grayscale image."""
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
        
        # Calculate statistics
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # Calculate entropy (measure of randomness)
        prob_dist = hist / np.sum(hist)
        entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
        
        # Calculate uniformity (chi-square test statistic)
        expected_freq = np.sum(hist) / 256
        chi_square = np.sum((hist - expected_freq) ** 2 / expected_freq)
        
        return {
            'type': 'grayscale',
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity),
            'entropy': float(entropy),
            'chi_square': float(chi_square),
            'histogram': hist.tolist(),
            'dynamic_range': float(np.max(image) - np.min(image)),
            'unique_values': int(len(np.unique(image))),
            'min_value': int(np.min(image)),
            'max_value': int(np.max(image))
        }
    
    def _analyze_color_histogram(self, image: np.ndarray) -> Dict:
        """Analyze histogram for color image."""
        results = {
            'type': 'color',
            'channels': {}
        }
        
        channel_names = ['red', 'green', 'blue']
        for i in range(min(3, image.shape[2])):
            channel = image[:, :, i]
            hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 255))
            
            mean_intensity = np.mean(channel)
            std_intensity = np.std(channel)
            
            # Calculate entropy
            prob_dist = hist / np.sum(hist)
            entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
            
            # Calculate uniformity
            expected_freq = np.sum(hist) / 256
            chi_square = np.sum((hist - expected_freq) ** 2 / expected_freq)
            
            results['channels'][channel_names[i]] = {
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'entropy': float(entropy),
                'chi_square': float(chi_square),
                'histogram': hist.tolist(),
                'dynamic_range': float(np.max(channel) - np.min(channel)),
                'unique_values': int(len(np.unique(channel))),
                'min_value': int(np.min(channel)),
                'max_value': int(np.max(channel))
            }
        
        # Overall image statistics
        results['overall'] = {
            'mean_intensity': float(np.mean(image)),
            'std_intensity': float(np.std(image)),
            'total_pixels': int(image.size)
        }
        
        # Calculate correlation coefficients between channels
        if image.shape[2] >= 2:
            results['overall']['correlation_rg'] = float(
                np.corrcoef(image[:,:,0].flatten(), image[:,:,1].flatten())[0,1]
            )
        if image.shape[2] >= 3:
            results['overall']['correlation_rb'] = float(
                np.corrcoef(image[:,:,0].flatten(), image[:,:,2].flatten())[0,1]
            )
            results['overall']['correlation_gb'] = float(
                np.corrcoef(image[:,:,1].flatten(), image[:,:,2].flatten())[0,1]
            )
        
        return results
    
    def analyze_compression_efficiency(self, original_image: np.ndarray, 
                                     compressed_data: Dict) -> Dict:
        """
        Analyze compression efficiency metrics.
        
        Args:
            original_image: Original image array
            compressed_data: Compressed data dictionary
            
        Returns:
            Dictionary with compression efficiency metrics
        """
        original_size = original_image.nbytes
        
        # Calculate compressed size
        compressed_size = 0
        if 'quantized_coefficients' in compressed_data:
            coeffs = compressed_data['quantized_coefficients']
            if isinstance(coeffs, list):  # Multi-channel
                for coeff_bytes in coeffs:
                    compressed_size += len(coeff_bytes)
            else:  # Single channel
                compressed_size += len(coeffs)
        
        # Add metadata overhead
        if 'quantization_matrix' in compressed_data:
            compressed_size += compressed_data['quantization_matrix'].nbytes
        compressed_size += 512  # Estimated metadata overhead
        
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        space_saved_percent = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        # Calculate bits per pixel
        total_pixels = original_image.shape[0] * original_image.shape[1]
        bpp = (compressed_size * 8) / total_pixels
        
        return {
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_ratio': compression_ratio,
            'space_saved_percent': space_saved_percent,
            'bits_per_pixel': bpp,
            'compression_efficiency': compression_ratio / (compressed_data.get('quality', 75) / 100.0)
        }
    
    def get_timing_report(self) -> Dict:
        """
        Get comprehensive timing report.
        
        Returns:
            Dictionary containing all timing measurements
        """
        total_time = sum(self.timing_data.values())
        
        report = {
            'total_time': total_time,
            'operations': self.timing_data.copy()
        }
        
        # Calculate percentages
        if total_time > 0:
            report['percentages'] = {
                op: (time_val / total_time) * 100 
                for op, time_val in self.timing_data.items()
            }
        else:
            report['percentages'] = {}
        
        return report
    
    def evaluate_encryption_strength(self, npcr: float, uaci: float) -> Dict:
        """
        Evaluate encryption strength based on NPCR and UACI values.
        
        Args:
            npcr: Number of Pixels Change Rate
            uaci: Unified Average Changing Intensity
            
        Returns:
            Dictionary with encryption strength evaluation
        """
        # Ideal values for good encryption
        ideal_npcr = 99.6094  # Theoretical ideal for good encryption
        ideal_uaci = 33.4635  # Theoretical ideal for good encryption
        
        # Calculate deviations
        npcr_deviation = abs(npcr - ideal_npcr)
        uaci_deviation = abs(uaci - ideal_uaci)
        
        # Determine strength levels
        if npcr >= 99.0 and 30.0 <= uaci <= 37.0:
            strength = "Excellent"
        elif npcr >= 95.0 and 25.0 <= uaci <= 40.0:
            strength = "Good"
        elif npcr >= 90.0 and 20.0 <= uaci <= 45.0:
            strength = "Fair"
        else:
            strength = "Poor"
        
        return {
            'strength_level': strength,
            'npcr_score': npcr,
            'uaci_score': uaci,
            'npcr_deviation_from_ideal': npcr_deviation,
            'uaci_deviation_from_ideal': uaci_deviation,
            'ideal_npcr': ideal_npcr,
            'ideal_uaci': ideal_uaci,
            'meets_security_threshold': npcr >= 99.0 and 30.0 <= uaci <= 37.0
        }
    
    def _resize_to_match(self, image: np.ndarray, target_shape: Tuple) -> np.ndarray:
        """
        Resize image to match target shape for comparison by cropping or padding.
        
        Args:
            image: Image to resize
            target_shape: Target shape to match
            
        Returns:
            Resized image
        """
        if image.shape == target_shape:
            return image
        
        # Handle channel conversion first
        if len(target_shape) == 2:
            # Target is grayscale
            if len(image.shape) == 3:
                # Convert color to grayscale
                image = np.mean(image, axis=2).astype(np.uint8)
        elif len(target_shape) == 3:
            # Target is color
            if len(image.shape) == 2:
                # Convert grayscale to RGB by stacking
                image = np.stack([image] * target_shape[2], axis=2)
            elif image.shape[2] != target_shape[2]:
                # Handle channel mismatch (e.g., RGBA to RGB)
                if target_shape[2] == 3 and image.shape[2] > 3:
                    # Take first 3 channels
                    image = image[:, :, :3]
                elif target_shape[2] > image.shape[2]:
                    # Pad channels by repeating the last channel
                    channels_needed = target_shape[2] - image.shape[2]
                    last_channel = image[:, :, -1:] if image.shape[2] > 0 else np.zeros((image.shape[0], image.shape[1], 1), dtype=image.dtype)
                    padding = np.repeat(last_channel, channels_needed, axis=2)
                    image = np.concatenate([image, padding], axis=2)
        
        # Now handle spatial dimensions
        if len(target_shape) == 2:
            target_h, target_w = target_shape
            current_h, current_w = image.shape[:2]
        else:
            target_h, target_w = target_shape[:2]
            current_h, current_w = image.shape[:2]
        
        # Crop or pad to match dimensions
        # First handle height
        if current_h > target_h:
            # Crop height (center crop)
            start_h = (current_h - target_h) // 2
            if len(image.shape) == 2:
                image = image[start_h:start_h + target_h, :]
            else:
                image = image[start_h:start_h + target_h, :, :]
        elif current_h < target_h:
            # Pad height
            pad_h = target_h - current_h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            if len(image.shape) == 2:
                image = np.pad(image, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
            else:
                image = np.pad(image, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=0)
        
        # Then handle width
        current_h, current_w = image.shape[:2]
        if current_w > target_w:
            # Crop width (center crop)
            start_w = (current_w - target_w) // 2
            if len(image.shape) == 2:
                image = image[:, start_w:start_w + target_w]
            else:
                image = image[:, start_w:start_w + target_w, :]
        elif current_w < target_w:
            # Pad width
            pad_w = target_w - current_w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            if len(image.shape) == 2:
                image = np.pad(image, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
            else:
                image = np.pad(image, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        
        return image
    
    def reset_timing(self):
        """Reset all timing data."""
        self.timing_data.clear()
    
    def generate_performance_report(self, original_image: np.ndarray, 
                                  compressed_data: Dict,
                                  reconstructed_image: np.ndarray,
                                  encrypted_visualization: np.ndarray) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            original_image: Original image array
            compressed_data: Compressed data dictionary
            reconstructed_image: Reconstructed image after decompression
            encrypted_visualization: Encrypted data visualization
            
        Returns:
            Complete performance analysis report
        """
        # Calculate all metrics
        psnr = self.calculate_psnr(original_image, reconstructed_image)
        npcr = self.calculate_npcr(original_image, encrypted_visualization)
        uaci = self.calculate_uaci(original_image, encrypted_visualization)
        
        # Analyze histograms
        original_histogram = self.analyze_histogram(original_image)
        encrypted_histogram = self.analyze_histogram(encrypted_visualization)
        
        # Compression efficiency
        compression_metrics = self.analyze_compression_efficiency(original_image, compressed_data)
        
        # Encryption strength evaluation
        encryption_strength = self.evaluate_encryption_strength(npcr, uaci)
        
        # Timing report
        timing_report = self.get_timing_report()
        
        return {
            'quality_metrics': {
                'psnr': psnr,
                'mse': ((255.0 / (10 ** (psnr / 20))) ** 2) if psnr != float('inf') else 0.0
            },
            'encryption_metrics': {
                'npcr': npcr,
                'uaci': uaci,
                'strength_evaluation': encryption_strength
            },
            'compression_metrics': compression_metrics,
            'histogram_analysis': {
                'original': original_histogram,
                'encrypted': encrypted_histogram
            },
            'timing_analysis': timing_report,
            'summary': {
                'overall_quality': 'Excellent' if psnr > 40 else 'Good' if psnr > 30 else 'Fair' if psnr > 20 else 'Poor',
                'encryption_security': encryption_strength['strength_level'],
                'compression_efficiency': 'High' if compression_metrics['compression_ratio'] > 5 else 'Medium' if compression_metrics['compression_ratio'] > 2 else 'Low'
            }
        }


# Convenience functions for direct usage
def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate PSNR directly."""
    analyzer = PerformanceAnalyzer()
    return analyzer.calculate_psnr(original, reconstructed)

def calculate_npcr(original: np.ndarray, encrypted: np.ndarray) -> float:
    """Calculate NPCR directly."""
    analyzer = PerformanceAnalyzer()
    return analyzer.calculate_npcr(original, encrypted)

def calculate_uaci(original: np.ndarray, encrypted: np.ndarray) -> float:
    """Calculate UACI directly."""
    analyzer = PerformanceAnalyzer()
    return analyzer.calculate_uaci(original, encrypted)


