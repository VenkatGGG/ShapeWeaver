"""
Data Loader for ShapeWeaver Training

This module handles loading and preprocessing of the synthetic dataset
for training the ShapeWeaver model.
"""

import tensorflow as tf
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image
import random


class ShapeWeaverDataLoader:
    """Data loader for ShapeWeaver dataset."""
    
    def __init__(self, dataset_dir, image_size=256, max_sequence_length=30000):
        """
        Initialize the data loader.
        
        Args:
            dataset_dir (str): Path to dataset directory
            image_size (int): Size of input images
            max_sequence_length (int): Maximum sequence length to handle
        """
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.max_sequence_length = max_sequence_length
        
        # Load dataset metadata
        metadata_path = self.dataset_dir / 'dataset_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None
        
        # Get all sample files
        self.sample_files = self._get_sample_files()
        print(f"Found {len(self.sample_files)} samples in dataset")
    
    def _get_sample_files(self):
        """Get list of all sample files in the dataset."""
        curve_files = list(self.dataset_dir.glob('curve_*.json'))
        
        samples = []
        for curve_file in curve_files:
            # Extract sample ID from filename
            sample_id = curve_file.stem.split('_')[1]
            image_file = self.dataset_dir / f'shape_{sample_id}.png'
            
            if image_file.exists():
                samples.append({
                    'sample_id': sample_id,
                    'image_path': str(image_file),
                    'curve_path': str(curve_file)
                })
        
        return samples
    
    def load_single_sample(self, sample_info):
        """
        Load a single sample (image + curve).
        
        Args:
            sample_info (dict): Sample information with paths
            
        Returns:
            tuple: (image_array, curve_sequence, metadata)
        """
        # Load image
        image = Image.open(sample_info['image_path']).convert('L')
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
        
        # Load curve data
        with open(sample_info['curve_path'], 'r') as f:
            curve_data = json.load(f)
        
        # Extract normalized sequence
        curve_sequence = np.array(curve_data['normalized_sequence'], dtype=np.float32)
        
        # Reshape to (sequence_length, 2) for (x, y) pairs
        curve_sequence = curve_sequence.reshape(-1, 2)
        
        return image_array, curve_sequence, curve_data['metadata']
    
    def create_tensorflow_dataset(self, batch_size=32, shuffle=True, 
                                split_ratio=0.8, seed=42):
        """
        Create TensorFlow datasets for training and validation.
        
        Args:
            batch_size (int): Batch size for training
            shuffle (bool): Whether to shuffle the data
            split_ratio (float): Ratio for train/validation split
            seed (int): Random seed for reproducibility
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        # Shuffle samples
        if shuffle:
            random.seed(seed)
            random.shuffle(self.sample_files)
        
        # Split into train/validation
        split_idx = int(len(self.sample_files) * split_ratio)
        train_samples = self.sample_files[:split_idx]
        val_samples = self.sample_files[split_idx:]
        
        print(f"Training samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples)}")
        
        # Create datasets
        train_dataset = self._create_dataset_from_samples(train_samples, batch_size, shuffle=True)
        val_dataset = self._create_dataset_from_samples(val_samples, batch_size, shuffle=False)
        
        return train_dataset, val_dataset
    
    def _create_dataset_from_samples(self, samples, batch_size, shuffle=True):
        """Create a TensorFlow dataset from sample list."""
        
        def generator():
            """Generator function for the dataset."""
            sample_indices = list(range(len(samples)))
            if shuffle:
                random.shuffle(sample_indices)
            
            for idx in sample_indices:
                try:
                    image, curve_sequence, metadata = self.load_single_sample(samples[idx])
                    
                    # Pad or truncate sequence to fixed length
                    processed_sequence = self._process_sequence(curve_sequence)
                    
                    yield image, processed_sequence
                    
                except Exception as e:
                    print(f"Error loading sample {samples[idx]['sample_id']}: {e}")
                    continue
        
        # Define output signature
        output_signature = (
            tf.TensorSpec(shape=(self.image_size, self.image_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(self.max_sequence_length, 2), dtype=tf.float32)
        )
        
        # Create dataset
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _process_sequence(self, curve_sequence):
        """
        Process curve sequence to fixed length.
        
        Args:
            curve_sequence (np.ndarray): Original curve sequence
            
        Returns:
            np.ndarray: Processed sequence of fixed length
        """
        seq_len = len(curve_sequence)
        
        if seq_len >= self.max_sequence_length:
            # Truncate if too long
            processed = curve_sequence[:self.max_sequence_length]
        else:
            # Pad if too short (pad with last coordinate)
            padding_len = self.max_sequence_length - seq_len
            last_coord = curve_sequence[-1:] if len(curve_sequence) > 0 else np.array([[0.5, 0.5]])
            padding = np.tile(last_coord, (padding_len, 1))
            processed = np.concatenate([curve_sequence, padding], axis=0)
        
        return processed.astype(np.float32)
    
    def get_dataset_statistics(self):
        """Get statistics about the dataset."""
        if not self.sample_files:
            return {}
        
        # Sample a few files to get statistics
        sample_size = min(100, len(self.sample_files))
        sampled_files = random.sample(self.sample_files, sample_size)
        
        sequence_lengths = []
        shape_types = []
        
        for sample_info in sampled_files:
            try:
                _, curve_sequence, metadata = self.load_single_sample(sample_info)
                sequence_lengths.append(len(curve_sequence))
                
                # Load curve data to get shape type
                with open(sample_info['curve_path'], 'r') as f:
                    curve_data = json.load(f)
                shape_types.append(curve_data.get('shape_type', 'unknown'))
                
            except Exception:
                continue
        
        return {
            'total_samples': len(self.sample_files),
            'sampled_size': len(sequence_lengths),
            'avg_sequence_length': np.mean(sequence_lengths) if sequence_lengths else 0,
            'max_sequence_length': np.max(sequence_lengths) if sequence_lengths else 0,
            'min_sequence_length': np.min(sequence_lengths) if sequence_lengths else 0,
            'shape_type_distribution': {
                shape_type: shape_types.count(shape_type) 
                for shape_type in set(shape_types)
            } if shape_types else {}
        }
    
    def visualize_sample(self, sample_idx=None):
        """
        Visualize a sample from the dataset.
        
        Args:
            sample_idx (int): Index of sample to visualize (random if None)
        """
        import matplotlib.pyplot as plt
        
        if sample_idx is None:
            sample_idx = random.randint(0, len(self.sample_files) - 1)
        
        sample_info = self.sample_files[sample_idx]
        image, curve_sequence, metadata = self.load_single_sample(sample_info)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(image.squeeze(), cmap='gray')
        ax1.set_title(f'Shape Image\nSample {sample_info["sample_id"]}')
        ax1.axis('off')
        
        # Curve points
        curve_image = np.zeros((self.image_size, self.image_size))
        for x_norm, y_norm in curve_sequence:
            x = int(x_norm * (self.image_size - 1))
            y = int(y_norm * (self.image_size - 1))
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                curve_image[y, x] = 1
        
        ax2.imshow(curve_image, cmap='hot')
        ax2.set_title(f'Curve Points\n{len(curve_sequence)} points')
        ax2.axis('off')
        
        # Combined view
        combined = image.squeeze() * 0.3 + curve_image * 0.7
        ax3.imshow(combined, cmap='viridis')
        ax3.set_title('Combined View')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print metadata
        print(f"Sample {sample_info['sample_id']} metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")


def test_data_loader():
    """Test the data loader functionality."""
    # Test with the generated dataset
    dataset_dir = "dataset"  # Adjust path as needed
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found. Please generate dataset first.")
        return
    
    # Create data loader
    data_loader = ShapeWeaverDataLoader(dataset_dir, max_sequence_length=1000)
    
    # Get statistics
    stats = data_loader.get_dataset_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test loading a single sample
    if data_loader.sample_files:
        sample_info = data_loader.sample_files[0]
        image, curve_sequence, metadata = data_loader.load_single_sample(sample_info)
        print(f"\nSample test:")
        print(f"  Image shape: {image.shape}")
        print(f"  Curve sequence shape: {curve_sequence.shape}")
        print(f"  Original sequence length: {metadata['curve_length']}")
    
    # Test dataset creation
    try:
        train_dataset, val_dataset = data_loader.create_tensorflow_dataset(
            batch_size=4
        )
        
        print(f"\nDataset creation test:")
        
        # Test one batch
        for batch_images, batch_sequences in train_dataset.take(1):
            print(f"  Batch images shape: {batch_images.shape}")
            print(f"  Batch sequences shape: {batch_sequences.shape}")
            break
        
    except Exception as e:
        print(f"Error creating dataset: {e}")


if __name__ == "__main__":
    test_data_loader()
