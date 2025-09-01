"""
Dataset Generation Script for ShapeWeaver

This script generates the complete synthetic dataset by creating thousands of
shape-curve pairs for training the ShapeWeaver model.
"""

import os
import json
import argparse
import random
from tqdm import tqdm
from pathlib import Path

from shape_generator import ShapeGenerator
from hilbert_curve import HilbertCurveGenerator


class DatasetGenerator:
    """Main class for generating the ShapeWeaver dataset."""
    
    def __init__(self, output_dir="dataset", image_size=256):
        """
        Initialize the dataset generator.
        
        Args:
            output_dir (str): Directory to save the dataset
            image_size (int): Size of generated images (must be power of 2)
        """
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize generators
        self.shape_generator = ShapeGenerator(image_size=image_size)
        self.hilbert_generator = HilbertCurveGenerator(image_size=image_size)
        
        # Shape type distribution for variety
        self.shape_types = {
            'random': 0.4,      # 40% random polygons
            'convex': 0.3,      # 30% convex polygons
            'circle': 0.15,     # 15% circles
            'ellipse': 0.15     # 15% ellipses
        }
    
    def generate_single_sample(self, sample_id):
        """
        Generate a single shape-curve pair.
        
        Args:
            sample_id (int): Unique identifier for this sample
            
        Returns:
            dict: Sample metadata and statistics
        """
        # Choose shape type based on distribution
        shape_type = random.choices(
            list(self.shape_types.keys()),
            weights=list(self.shape_types.values())
        )[0]
        
        # Generate shape mask
        mask = self.shape_generator.generate_random_shape_mask(shape_type)
        
        # Generate corresponding curve
        curve_points = self.hilbert_generator.generate_curve_for_shape(mask)
        
        # Convert to normalized sequence
        curve_sequence = self.hilbert_generator.curve_to_sequence(curve_points)
        
        # Create file paths
        image_path = self.output_dir / f"shape_{sample_id:06d}.png"
        curve_path = self.output_dir / f"curve_{sample_id:06d}.json"
        
        # Save shape image
        self.shape_generator.save_mask_as_image(mask, str(image_path))
        
        # Save curve data
        curve_data = {
            'sample_id': sample_id,
            'shape_type': shape_type,
            'image_path': str(image_path.name),
            'curve_points': curve_points,
            'normalized_sequence': curve_sequence,
            'metadata': self.hilbert_generator.calculate_curve_statistics(curve_points, mask)
        }
        
        with open(curve_path, 'w') as f:
            json.dump(curve_data, f, indent=2)
        
        return curve_data['metadata']
    
    def generate_dataset(self, num_samples, start_id=0, show_progress=True):
        """
        Generate the complete dataset.
        
        Args:
            num_samples (int): Number of samples to generate
            start_id (int): Starting sample ID (for resuming generation)
            show_progress (bool): Whether to show progress bar
            
        Returns:
            dict: Dataset statistics
        """
        print(f"Generating {num_samples} samples in {self.output_dir}")
        print(f"Image size: {self.image_size}x{self.image_size}")
        print(f"Shape type distribution: {self.shape_types}")
        
        # Statistics tracking
        all_stats = []
        
        # Generate samples with progress bar
        iterator = range(start_id, start_id + num_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating samples")
        
        for sample_id in iterator:
            try:
                stats = self.generate_single_sample(sample_id)
                all_stats.append(stats)
                
                # Update progress bar with current stats
                if show_progress and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({
                        'avg_coverage': f"{sum(s['coverage_percentage'] for s in all_stats)/len(all_stats):.1f}%",
                        'avg_points': f"{sum(s['curve_length'] for s in all_stats)/len(all_stats):.0f}"
                    })
                    
            except Exception as e:
                print(f"Error generating sample {sample_id}: {e}")
                continue
        
        # Calculate and save dataset statistics
        dataset_stats = self._calculate_dataset_statistics(all_stats)
        self._save_dataset_metadata(dataset_stats, num_samples)
        
        print(f"\nDataset generation complete!")
        self._print_dataset_summary(dataset_stats)
        
        return dataset_stats
    
    def _calculate_dataset_statistics(self, all_stats):
        """Calculate overall dataset statistics."""
        if not all_stats:
            return {}
        
        import numpy as np
        
        # Extract metrics
        coverages = [s['coverage_percentage'] for s in all_stats]
        curve_lengths = [s['curve_length'] for s in all_stats]
        shape_pixels = [s['total_shape_pixels'] for s in all_stats]
        densities = [s['curve_density'] for s in all_stats]
        
        return {
            'total_samples': len(all_stats),
            'image_size': self.image_size,
            'coverage_stats': {
                'mean': float(np.mean(coverages)),
                'std': float(np.std(coverages)),
                'min': float(np.min(coverages)),
                'max': float(np.max(coverages)),
                'median': float(np.median(coverages))
            },
            'curve_length_stats': {
                'mean': float(np.mean(curve_lengths)),
                'std': float(np.std(curve_lengths)),
                'min': float(np.min(curve_lengths)),
                'max': float(np.max(curve_lengths)),
                'median': float(np.median(curve_lengths))
            },
            'shape_size_stats': {
                'mean': float(np.mean(shape_pixels)),
                'std': float(np.std(shape_pixels)),
                'min': float(np.min(shape_pixels)),
                'max': float(np.max(shape_pixels)),
                'median': float(np.median(shape_pixels))
            },
            'density_stats': {
                'mean': float(np.mean(densities)),
                'std': float(np.std(densities)),
                'min': float(np.min(densities)),
                'max': float(np.max(densities)),
                'median': float(np.median(densities))
            }
        }
    
    def _save_dataset_metadata(self, stats, num_samples):
        """Save dataset metadata and statistics."""
        metadata = {
            'dataset_info': {
                'name': 'ShapeWeaver Synthetic Dataset',
                'version': '1.0',
                'description': 'Synthetic dataset of shape masks and space-filling curves',
                'total_samples': num_samples,
                'image_size': self.image_size,
                'shape_types': self.shape_types
            },
            'statistics': stats,
            'file_format': {
                'images': 'PNG files (256x256 grayscale, binary masks)',
                'curves': 'JSON files with curve points and metadata',
                'naming': 'shape_XXXXXX.png / curve_XXXXXX.json'
            }
        }
        
        metadata_path = self.output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset metadata saved to {metadata_path}")
    
    def _print_dataset_summary(self, stats):
        """Print a summary of the generated dataset."""
        print(f"Dataset Summary:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Image size: {stats['image_size']}x{stats['image_size']}")
        print(f"  Average coverage: {stats['coverage_stats']['mean']:.1f}% ± {stats['coverage_stats']['std']:.1f}%")
        print(f"  Average curve length: {stats['curve_length_stats']['mean']:.0f} ± {stats['curve_length_stats']['std']:.0f} points")
        print(f"  Average shape size: {stats['shape_size_stats']['mean']:.0f} ± {stats['shape_size_stats']['std']:.0f} pixels")
        print(f"  Curve length range: {stats['curve_length_stats']['min']:.0f} - {stats['curve_length_stats']['max']:.0f} points")
    
    def validate_dataset(self, num_samples_to_check=10):
        """
        Validate a subset of the generated dataset.
        
        Args:
            num_samples_to_check (int): Number of random samples to validate
        """
        print(f"Validating {num_samples_to_check} random samples...")
        
        # Get all curve files
        curve_files = list(self.output_dir.glob("curve_*.json"))
        
        if len(curve_files) == 0:
            print("No dataset files found to validate.")
            return
        
        # Check random samples
        sample_files = random.sample(curve_files, min(num_samples_to_check, len(curve_files)))
        
        for curve_file in sample_files:
            try:
                # Load curve data
                with open(curve_file, 'r') as f:
                    curve_data = json.load(f)
                
                # Check corresponding image exists
                image_file = self.output_dir / curve_data['image_path']
                if not image_file.exists():
                    print(f"❌ Missing image file: {image_file}")
                    continue
                
                # Load and validate image
                from PIL import Image
                img = Image.open(image_file)
                if img.size != (self.image_size, self.image_size):
                    print(f"❌ Wrong image size: {img.size}, expected {(self.image_size, self.image_size)}")
                    continue
                
                # Validate curve data structure
                required_keys = ['sample_id', 'shape_type', 'curve_points', 'normalized_sequence', 'metadata']
                if not all(key in curve_data for key in required_keys):
                    print(f"❌ Missing keys in {curve_file}")
                    continue
                
                print(f"✅ {curve_file.name}: {len(curve_data['curve_points'])} points, {curve_data['shape_type']} shape")
                
            except Exception as e:
                print(f"❌ Error validating {curve_file}: {e}")
        
        print("Validation complete.")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Generate ShapeWeaver synthetic dataset')
    parser.add_argument('--num_samples', type=int, default=1000, 
                       help='Number of samples to generate (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='Output directory for dataset (default: dataset)')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Size of generated images (default: 256)')
    parser.add_argument('--start_id', type=int, default=0,
                       help='Starting sample ID for resuming generation (default: 0)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset after generation')
    parser.add_argument('--quick_test', action='store_true',
                       help='Generate a small test dataset (10 samples)')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.num_samples = 10
        args.output_dir = 'test_dataset'
        print("Quick test mode: generating 10 samples in test_dataset/")
    
    # Create dataset generator
    generator = DatasetGenerator(
        output_dir=args.output_dir,
        image_size=args.image_size
    )
    
    # Generate dataset
    generator.generate_dataset(
        num_samples=args.num_samples,
        start_id=args.start_id
    )
    
    # Validate if requested
    if args.validate:
        generator.validate_dataset()


if __name__ == "__main__":
    main()
