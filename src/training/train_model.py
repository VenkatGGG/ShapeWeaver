"""
Training Pipeline for ShapeWeaver Model

This script handles the complete training process for the ShapeWeaver model,
including data loading, model compilation, training, and evaluation.
"""

import tensorflow as tf
import numpy as np
import argparse
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directories to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.shapeweaver_model import create_shapeweaver_model
from model.data_loader import ShapeWeaverDataLoader


class ShapeWeaverTrainer:
    """Main trainer class for ShapeWeaver model."""
    
    def __init__(self, dataset_dir, model_save_dir="models", log_dir="logs"):
        """
        Initialize the trainer.
        
        Args:
            dataset_dir (str): Path to dataset directory
            model_save_dir (str): Directory to save trained models
            log_dir (str): Directory for training logs
        """
        self.dataset_dir = dataset_dir
        self.model_save_dir = Path(model_save_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.model_save_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize data loader
        self.data_loader = ShapeWeaverDataLoader(dataset_dir)
        
        # Get dataset statistics to determine optimal sequence length
        stats = self.data_loader.get_dataset_statistics()
        
        # Use a reasonable sequence length for training (can be adjusted)
        # Too long sequences cause memory issues during training
        self.max_sequence_length = min(int(stats['avg_sequence_length']), 5000)
        print(f"Using max sequence length: {self.max_sequence_length}")
        
        # Update data loader with optimal sequence length
        self.data_loader.max_sequence_length = self.max_sequence_length
        
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.history = None
    
    def prepare_data(self, batch_size=8, validation_split=0.2):
        """
        Prepare training and validation datasets.
        
        Args:
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data for validation
        """
        print("Preparing datasets...")
        
        self.train_dataset, self.val_dataset = self.data_loader.create_tensorflow_dataset(
            batch_size=batch_size,
            split_ratio=1.0 - validation_split,
            shuffle=True
        )
        
        print("Datasets prepared successfully")
    
    def create_model(self, latent_dim=512, lstm_units=256):
        """
        Create and compile the ShapeWeaver model.
        
        Args:
            latent_dim (int): Dimensionality of the latent space
            lstm_units (int): Number of LSTM units in decoder
        """
        print("Creating model...")
        
        self.model = create_shapeweaver_model(
            latent_dim=latent_dim,
            lstm_units=lstm_units,
            max_sequence_length=self.max_sequence_length
        )
        
        # Custom loss function for sequence prediction
        def sequence_loss(y_true, y_pred):
            """Custom loss function for sequence prediction."""
            # Mean squared error for coordinate prediction
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            return mse
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss=sequence_loss,
            metrics=['mae']
        )
        
        # Print model summary
        summary = self.model.get_model_summary()
        print("Model Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        return self.model
    
    def create_callbacks(self, experiment_name=None):
        """Create training callbacks."""
        if experiment_name is None:
            experiment_name = f"shapeweaver_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_dir = self.log_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        callbacks = [
            # Model checkpointing
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(experiment_dir / 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir=str(experiment_dir / 'tensorboard'),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks, experiment_dir
    
    def train(self, epochs=50, batch_size=8, latent_dim=512, lstm_units=256, 
              experiment_name=None):
        """
        Train the ShapeWeaver model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            latent_dim (int): Latent space dimensionality
            lstm_units (int): LSTM units in decoder
            experiment_name (str): Name for this training experiment
        """
        print(f"Starting training for {epochs} epochs...")
        
        # Prepare data
        self.prepare_data(batch_size=batch_size)
        
        # Create model
        self.create_model(latent_dim=latent_dim, lstm_units=lstm_units)
        
        # Create callbacks
        callbacks, experiment_dir = self.create_callbacks(experiment_name)
        
        # Prepare data format for training
        def format_data_for_training(dataset):
            """Format dataset for model training."""
            def process_batch(images, sequences):
                # Create inputs dictionary
                inputs = {
                    'image': images,
                    'target_sequence': sequences
                }
                # Target is the same sequence (teacher forcing)
                targets = sequences
                return inputs, targets
            
            return dataset.map(process_batch)
        
        train_formatted = format_data_for_training(self.train_dataset)
        val_formatted = format_data_for_training(self.val_dataset)
        
        # Start training
        print("Training started...")
        self.history = self.model.fit(
            train_formatted,
            validation_data=val_formatted,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = experiment_dir / 'final_model.h5'
        self.model.save(str(final_model_path))
        print(f"Final model saved to {final_model_path}")
        
        # Save training history
        history_path = experiment_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {}
            for key, values in self.history.history.items():
                history_dict[key] = [float(v) for v in values]
            json.dump(history_dict, f, indent=2)
        
        print(f"Training completed! Results saved in {experiment_dir}")
        
        return self.history
    
    def evaluate_model(self, model_path=None):
        """
        Evaluate the trained model.
        
        Args:
            model_path (str): Path to saved model (uses current model if None)
        """
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        
        if self.model is None:
            print("No model available for evaluation")
            return
        
        if self.val_dataset is None:
            print("No validation dataset available")
            return
        
        print("Evaluating model...")
        
        # Evaluate on validation set
        val_formatted = self.val_dataset.map(
            lambda images, sequences: (
                {'image': images, 'target_sequence': sequences}, 
                sequences
            )
        )
        
        results = self.model.evaluate(val_formatted, verbose=1)
        
        print(f"Validation Results:")
        print(f"  Loss: {results[0]:.4f}")
        print(f"  MAE: {results[1]:.4f}")
        
        return results
    
    def visualize_training_history(self, save_path=None):
        """Visualize training history."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()
    
    def test_single_prediction(self, sample_idx=0):
        """
        Test the model on a single sample.
        
        Args:
            sample_idx (int): Index of sample to test
        """
        if self.model is None:
            print("No model available for testing")
            return
        
        # Load a single sample
        sample_info = self.data_loader.sample_files[sample_idx]
        image, true_sequence, metadata = self.data_loader.load_single_sample(sample_info)
        
        # Prepare for prediction
        image_batch = np.expand_dims(image, 0)
        
        # Generate prediction
        predicted_sequence = self.model.predict_curve(image_batch)
        predicted_sequence = predicted_sequence[0]  # Remove batch dimension
        
        # Visualize results
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(image.squeeze(), cmap='gray')
        ax1.set_title(f'Input Shape\nSample {sample_info["sample_id"]}')
        ax1.axis('off')
        
        # True curve
        true_curve_image = np.zeros((256, 256))
        for x_norm, y_norm in true_sequence[:1000]:  # Show first 1000 points
            x = int(x_norm * 255)
            y = int(y_norm * 255)
            if 0 <= x < 256 and 0 <= y < 256:
                true_curve_image[y, x] = 1
        
        ax2.imshow(true_curve_image, cmap='hot')
        ax2.set_title(f'True Curve\n{len(true_sequence)} points')
        ax2.axis('off')
        
        # Predicted curve
        pred_curve_image = np.zeros((256, 256))
        for x_norm, y_norm in predicted_sequence[:1000]:  # Show first 1000 points
            x = int(x_norm * 255)
            y = int(y_norm * 255)
            if 0 <= x < 256 and 0 <= y < 256:
                pred_curve_image[y, x] = 1
        
        ax3.imshow(pred_curve_image, cmap='hot')
        ax3.set_title(f'Predicted Curve\n{len(predicted_sequence)} points')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate metrics
        min_len = min(len(true_sequence), len(predicted_sequence))
        mse = np.mean((true_sequence[:min_len] - predicted_sequence[:min_len]) ** 2)
        mae = np.mean(np.abs(true_sequence[:min_len] - predicted_sequence[:min_len]))
        
        print(f"Prediction metrics for sample {sample_info['sample_id']}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  True sequence length: {len(true_sequence)}")
        print(f"  Predicted sequence length: {len(predicted_sequence)}")


def main():
    """Main function for command-line training."""
    parser = argparse.ArgumentParser(description='Train ShapeWeaver model')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=512,
                       help='Latent space dimensionality')
    parser.add_argument('--lstm_units', type=int, default=256,
                       help='LSTM units in decoder')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this training experiment')
    parser.add_argument('--evaluate_only', type=str, default=None,
                       help='Path to model for evaluation only')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ShapeWeaverTrainer(args.dataset_dir)
    
    if args.evaluate_only:
        # Only evaluate existing model
        trainer.evaluate_model(args.evaluate_only)
        trainer.test_single_prediction()
    else:
        # Train new model
        history = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            lstm_units=args.lstm_units,
            experiment_name=args.experiment_name
        )
        
        # Visualize results
        trainer.visualize_training_history()
        
        # Test on single sample
        trainer.test_single_prediction()


if __name__ == "__main__":
    main()
