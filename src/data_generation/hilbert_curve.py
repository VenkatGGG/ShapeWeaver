"""
Hilbert Curve Generator for ShapeWeaver

This module generates Hilbert space-filling curves and clips them to shape boundaries.
The Hilbert curve is chosen for its excellent locality-preserving properties.
"""

import numpy as np
import math
from typing import List, Tuple


class HilbertCurveGenerator:
    """Generates Hilbert curves and clips them to shape boundaries."""
    
    def __init__(self, image_size=256):
        """
        Initialize the Hilbert curve generator.
        
        Args:
            image_size (int): Size of the square grid (must be power of 2)
        """
        self.image_size = image_size
        
        # Find the order needed to cover the image
        # For 256x256, we need order 8 (2^8 = 256)
        self.order = int(math.log2(image_size))
        
        if 2 ** self.order != image_size:
            raise ValueError(f"Image size {image_size} must be a power of 2")
    
    def generate_hilbert_curve(self) -> List[Tuple[int, int]]:
        """
        Generate a complete Hilbert curve covering the entire grid.
        
        Returns:
            List[Tuple[int, int]]: Ordered list of (x, y) coordinates
        """
        points = []
        n = 2 ** self.order
        
        for i in range(n * n):
            x, y = self._hilbert_index_to_xy(i, self.order)
            points.append((x, y))
        
        return points
    
    def _hilbert_index_to_xy(self, index: int, order: int) -> Tuple[int, int]:
        """
        Convert Hilbert curve index to (x, y) coordinates.
        
        Args:
            index (int): Position along the Hilbert curve
            order (int): Order of the Hilbert curve
            
        Returns:
            Tuple[int, int]: (x, y) coordinates
        """
        x = y = 0
        t = index
        n = 2 ** order
        
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = self._hilbert_rotate(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        
        return x, y
    
    def _hilbert_rotate(self, n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
        """
        Rotate/flip coordinates for Hilbert curve generation.
        
        Args:
            n (int): Size of current quadrant
            x, y (int): Current coordinates
            rx, ry (int): Rotation flags
            
        Returns:
            Tuple[int, int]: Rotated (x, y) coordinates
        """
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        
        return x, y
    
    def clip_curve_to_mask(self, curve_points: List[Tuple[int, int]], 
                          mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Clip Hilbert curve to only include points inside the shape mask.
        
        Args:
            curve_points (List[Tuple[int, int]]): Full Hilbert curve points
            mask (np.ndarray): Binary mask where 1 = inside shape
            
        Returns:
            List[Tuple[int, int]]: Filtered curve points inside the shape
        """
        clipped_points = []
        
        for x, y in curve_points:
            # Check if point is within image bounds
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                # Check if point is inside the shape (mask value = 1)
                if mask[y, x] == 1:
                    clipped_points.append((x, y))
        
        return clipped_points
    
    def generate_curve_for_shape(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Generate a space-filling curve for a given shape mask.
        
        Args:
            mask (np.ndarray): Binary mask of the shape
            
        Returns:
            List[Tuple[int, int]]: Ordered curve points filling the shape
        """
        # Generate full Hilbert curve
        full_curve = self.generate_hilbert_curve()
        
        # Clip to shape
        clipped_curve = self.clip_curve_to_mask(full_curve, mask)
        
        return clipped_curve
    
    def curve_to_sequence(self, curve_points: List[Tuple[int, int]]) -> List[float]:
        """
        Convert curve points to a normalized sequence for ML training.
        
        Args:
            curve_points (List[Tuple[int, int]]): Curve points
            
        Returns:
            List[float]: Flattened, normalized sequence [x1, y1, x2, y2, ...]
        """
        sequence = []
        
        for x, y in curve_points:
            # Normalize coordinates to [0, 1] range
            norm_x = x / (self.image_size - 1)
            norm_y = y / (self.image_size - 1)
            sequence.extend([norm_x, norm_y])
        
        return sequence
    
    def sequence_to_curve(self, sequence: List[float]) -> List[Tuple[int, int]]:
        """
        Convert normalized sequence back to curve points.
        
        Args:
            sequence (List[float]): Normalized sequence [x1, y1, x2, y2, ...]
            
        Returns:
            List[Tuple[int, int]]: Curve points
        """
        points = []
        
        for i in range(0, len(sequence), 2):
            if i + 1 < len(sequence):
                norm_x, norm_y = sequence[i], sequence[i + 1]
                
                # Denormalize coordinates
                x = int(norm_x * (self.image_size - 1))
                y = int(norm_y * (self.image_size - 1))
                
                points.append((x, y))
        
        return points
    
    def visualize_curve_on_mask(self, mask: np.ndarray, curve_points: List[Tuple[int, int]]):
        """
        Visualize the generated curve overlaid on the shape mask.
        
        Args:
            mask (np.ndarray): Shape mask
            curve_points (List[Tuple[int, int]]): Curve points to visualize
        """
        import matplotlib.pyplot as plt
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original mask
        ax1.imshow(mask, cmap='gray')
        ax1.set_title('Original Shape Mask')
        ax1.axis('off')
        
        # Curve points only
        curve_image = np.zeros_like(mask)
        for x, y in curve_points:
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                curve_image[y, x] = 1
        
        ax2.imshow(curve_image, cmap='hot')
        ax2.set_title('Generated Curve')
        ax2.axis('off')
        
        # Combined view
        combined = mask.astype(float) * 0.3
        for x, y in curve_points:
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                combined[y, x] = 1.0
        
        ax3.imshow(combined, cmap='viridis')
        ax3.set_title('Curve on Shape')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_curve_statistics(self, curve_points: List[Tuple[int, int]], 
                                 mask: np.ndarray) -> dict:
        """
        Calculate statistics about the generated curve.
        
        Args:
            curve_points (List[Tuple[int, int]]): Generated curve points
            mask (np.ndarray): Original shape mask
            
        Returns:
            dict: Statistics about the curve
        """
        total_shape_pixels = np.sum(mask)
        curve_length = len(curve_points)
        
        # Calculate coverage percentage
        coverage = (curve_length / total_shape_pixels) * 100 if total_shape_pixels > 0 else 0
        
        # Calculate curve density (points per unit area)
        shape_area = total_shape_pixels
        density = curve_length / shape_area if shape_area > 0 else 0
        
        return {
            'total_shape_pixels': int(total_shape_pixels),
            'curve_length': curve_length,
            'coverage_percentage': coverage,
            'curve_density': density,
            'normalized_sequence_length': curve_length * 2  # x, y pairs
        }


def test_hilbert_generator():
    """Test function to verify Hilbert curve generation works correctly."""
    from shape_generator import ShapeGenerator
    
    # Create a test shape
    shape_gen = ShapeGenerator(image_size=256)
    mask = shape_gen.generate_random_shape_mask('convex')
    
    # Generate Hilbert curve
    hilbert_gen = HilbertCurveGenerator(image_size=256)
    curve_points = hilbert_gen.generate_curve_for_shape(mask)
    
    # Calculate statistics
    stats = hilbert_gen.calculate_curve_statistics(curve_points, mask)
    print("Curve Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test sequence conversion
    sequence = hilbert_gen.curve_to_sequence(curve_points)
    reconstructed_points = hilbert_gen.sequence_to_curve(sequence)
    
    print(f"\nSequence conversion test:")
    print(f"Original points: {len(curve_points)}")
    print(f"Sequence length: {len(sequence)}")
    print(f"Reconstructed points: {len(reconstructed_points)}")
    print(f"First 5 original: {curve_points[:5]}")
    print(f"First 5 reconstructed: {reconstructed_points[:5]}")
    
    # Visualize result
    hilbert_gen.visualize_curve_on_mask(mask, curve_points)


if __name__ == "__main__":
    test_hilbert_generator()
