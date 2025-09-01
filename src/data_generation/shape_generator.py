"""
Shape Generator for ShapeWeaver Dataset Creation

This module generates random polygons and converts them to binary image masks.
The polygons are created by generating random points and sorting them by angle
to ensure non-self-intersecting shapes.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw
import random
import math


class ShapeGenerator:
    """Generates random polygon shapes as binary image masks."""
    
    def __init__(self, image_size=256, min_vertices=3, max_vertices=12):
        """
        Initialize the shape generator.
        
        Args:
            image_size (int): Size of the output square image
            min_vertices (int): Minimum number of polygon vertices
            max_vertices (int): Maximum number of polygon vertices
        """
        self.image_size = image_size
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices
    
    def generate_random_polygon(self, center_x=None, center_y=None, min_radius=50, max_radius=100):
        """
        Generate a random polygon by creating points around a center and sorting by angle.
        
        Args:
            center_x (float): Center X coordinate (defaults to image center)
            center_y (float): Center Y coordinate (defaults to image center)
            min_radius (float): Minimum distance from center to vertices
            max_radius (float): Maximum distance from center to vertices
            
        Returns:
            list: List of (x, y) tuples representing polygon vertices
        """
        if center_x is None:
            center_x = self.image_size // 2
        if center_y is None:
            center_y = self.image_size // 2
            
        # Random number of vertices
        num_vertices = random.randint(self.min_vertices, self.max_vertices)
        
        vertices = []
        for _ in range(num_vertices):
            # Random angle and radius
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(min_radius, max_radius)
            
            # Calculate vertex position
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Ensure vertices stay within image bounds with some margin
            x = max(10, min(self.image_size - 10, x))
            y = max(10, min(self.image_size - 10, y))
            
            vertices.append((x, y))
        
        # Sort vertices by angle to create a proper polygon
        vertices = self._sort_vertices_by_angle(vertices, center_x, center_y)
        
        return vertices
    
    def _sort_vertices_by_angle(self, vertices, center_x, center_y):
        """
        Sort vertices by their angle relative to the center point.
        This ensures the polygon doesn't self-intersect.
        """
        def angle_from_center(vertex):
            x, y = vertex
            return math.atan2(y - center_y, x - center_x)
        
        return sorted(vertices, key=angle_from_center)
    
    def generate_convex_polygon(self, center_x=None, center_y=None, min_radius=50, max_radius=100):
        """
        Generate a convex polygon using a different approach.
        Creates more regular, convex shapes.
        """
        if center_x is None:
            center_x = self.image_size // 2
        if center_y is None:
            center_y = self.image_size // 2
            
        num_vertices = random.randint(self.min_vertices, self.max_vertices)
        
        # Generate angles evenly distributed with some randomness
        base_angles = [i * 2 * math.pi / num_vertices for i in range(num_vertices)]
        angles = [angle + random.uniform(-0.3, 0.3) for angle in base_angles]
        
        vertices = []
        for angle in angles:
            radius = random.uniform(min_radius, max_radius)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Ensure vertices stay within bounds
            x = max(10, min(self.image_size - 10, x))
            y = max(10, min(self.image_size - 10, y))
            
            vertices.append((x, y))
        
        return vertices
    
    def polygon_to_mask(self, vertices):
        """
        Convert polygon vertices to a binary image mask.
        
        Args:
            vertices (list): List of (x, y) tuples
            
        Returns:
            numpy.ndarray: Binary mask where 1 = inside polygon, 0 = outside
        """
        # Create blank image
        image = Image.new('L', (self.image_size, self.image_size), 0)
        draw = ImageDraw.Draw(image)
        
        # Draw filled polygon
        draw.polygon(vertices, fill=255)
        
        # Convert to numpy array and normalize to 0/1
        mask = np.array(image)
        mask = (mask > 127).astype(np.uint8)
        
        return mask
    
    def generate_random_shape_mask(self, shape_type='random'):
        """
        Generate a random shape mask.
        
        Args:
            shape_type (str): Type of shape ('random', 'convex', 'circle', 'ellipse')
            
        Returns:
            numpy.ndarray: Binary mask of the generated shape
        """
        if shape_type == 'random':
            vertices = self.generate_random_polygon()
        elif shape_type == 'convex':
            vertices = self.generate_convex_polygon()
        elif shape_type == 'circle':
            return self._generate_circle_mask()
        elif shape_type == 'ellipse':
            return self._generate_ellipse_mask()
        else:
            vertices = self.generate_random_polygon()
        
        return self.polygon_to_mask(vertices)
    
    def _generate_circle_mask(self):
        """Generate a circular mask."""
        center = self.image_size // 2
        radius = random.randint(40, 90)
        
        y, x = np.ogrid[:self.image_size, :self.image_size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        
        return mask.astype(np.uint8)
    
    def _generate_ellipse_mask(self):
        """Generate an elliptical mask."""
        center = self.image_size // 2
        a = random.randint(30, 80)  # Semi-major axis
        b = random.randint(30, 80)  # Semi-minor axis
        
        y, x = np.ogrid[:self.image_size, :self.image_size]
        mask = ((x - center) / a) ** 2 + ((y - center) / b) ** 2 <= 1
        
        return mask.astype(np.uint8)
    
    def save_mask_as_image(self, mask, filepath):
        """
        Save binary mask as PNG image.
        
        Args:
            mask (numpy.ndarray): Binary mask
            filepath (str): Output file path
        """
        # Convert to 0-255 range
        image_array = mask * 255
        image = Image.fromarray(image_array.astype(np.uint8), 'L')
        image.save(filepath)
    
    def visualize_mask(self, mask):
        """
        Display the mask for debugging purposes.
        
        Args:
            mask (numpy.ndarray): Binary mask to visualize
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(6, 6))
        plt.imshow(mask, cmap='gray')
        plt.title('Generated Shape Mask')
        plt.axis('off')
        plt.show()


def test_shape_generator():
    """Test function to verify shape generation works correctly."""
    generator = ShapeGenerator()
    
    # Test different shape types
    shape_types = ['random', 'convex', 'circle', 'ellipse']
    
    for i, shape_type in enumerate(shape_types):
        mask = generator.generate_random_shape_mask(shape_type)
        print(f"Generated {shape_type} shape: {mask.shape}, unique values: {np.unique(mask)}")
        
        # Save test image
        generator.save_mask_as_image(mask, f"/tmp/test_shape_{i}_{shape_type}.png")


if __name__ == "__main__":
    test_shape_generator()
