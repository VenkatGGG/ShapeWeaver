"""
ShapeWeaver Model Architecture

This module implements the CNN Encoder + LSTM Decoder architecture for
converting shape images to space-filling curve sequences.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class ShapeWeaverEncoder(keras.Model):
    """
    CNN Encoder that processes shape images into context vectors.
    
    The encoder takes a 256x256 binary image and progressively extracts
    features through convolutional layers, outputting a dense representation.
    """
    
    def __init__(self, latent_dim=512):
        """
        Initialize the encoder.
        
        Args:
            latent_dim (int): Dimensionality of the output context vector
        """
        super(ShapeWeaverEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Convolutional layers for feature extraction
        self.conv_layers = [
            # First block: 256x256 -> 128x128
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Second block: 128x128 -> 64x64
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Third block: 64x64 -> 32x32
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Fourth block: 32x32 -> 16x16
            layers.Conv2D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Fifth block: 16x16 -> 8x8
            layers.Conv2D(512, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Sixth block: 8x8 -> 4x4
            layers.Conv2D(512, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
        ]
        
        # Global pooling and dense layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(1024, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(latent_dim, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
    
    def call(self, inputs, training=None):
        """
        Forward pass through the encoder.
        
        Args:
            inputs: Shape images [batch_size, 256, 256, 1]
            training: Whether in training mode
            
        Returns:
            Context vectors [batch_size, latent_dim]
        """
        x = inputs
        
        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x, training=training)
        
        # Global pooling and dense layers
        x = self.global_pool(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        
        return x


class ShapeWeaverDecoder(keras.Model):
    """
    LSTM Decoder that generates curve sequences from context vectors.
    
    The decoder takes a context vector and generates an ordered sequence
    of (x, y) coordinates representing the space-filling curve.
    """
    
    def __init__(self, latent_dim=512, lstm_units=256, max_sequence_length=30000):
        """
        Initialize the decoder.
        
        Args:
            latent_dim (int): Dimensionality of input context vector
            lstm_units (int): Number of LSTM units
            max_sequence_length (int): Maximum curve sequence length
        """
        super(ShapeWeaverDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.max_sequence_length = max_sequence_length
        
        # Context vector processing
        self.context_dense = layers.Dense(lstm_units, activation='relu')
        
        # LSTM layers for sequence generation
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.lstm2 = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        
        # Output layers
        self.output_dense = layers.Dense(2, activation='sigmoid')  # (x, y) coordinates
        
        # Special tokens
        self.start_token = tf.constant([0.5, 0.5], dtype=tf.float32)  # Center of image
        self.end_token = tf.constant([1.0, 1.0], dtype=tf.float32)   # Special end marker
    
    def call(self, context_vector, target_sequence=None, training=None):
        """
        Forward pass through the decoder.
        
        Args:
            context_vector: Context from encoder [batch_size, latent_dim]
            target_sequence: Target coordinates for training [batch_size, seq_len, 2]
            training: Whether in training mode
            
        Returns:
            Generated sequence [batch_size, seq_len, 2]
        """
        batch_size = tf.shape(context_vector)[0]
        
        # Process context vector to initialize LSTM states
        context_processed = self.context_dense(context_vector)
        
        # Initialize LSTM states
        initial_state_h1 = context_processed
        initial_state_c1 = tf.zeros_like(context_processed)
        initial_state_h2 = tf.zeros_like(context_processed)
        initial_state_c2 = tf.zeros_like(context_processed)
        
        if training and target_sequence is not None:
            # Teacher forcing during training
            return self._teacher_forcing_forward(
                target_sequence, 
                [initial_state_h1, initial_state_c1],
                [initial_state_h2, initial_state_c2],
                training
            )
        else:
            # Autoregressive generation during inference
            return self._autoregressive_forward(
                batch_size,
                [initial_state_h1, initial_state_c1],
                [initial_state_h2, initial_state_c2],
                training
            )
    
    def _teacher_forcing_forward(self, target_sequence, state1, state2, training):
        """Teacher forcing forward pass for training."""
        # Prepend start token to target sequence
        batch_size = tf.shape(target_sequence)[0]
        start_tokens = tf.expand_dims(
            tf.tile(tf.expand_dims(self.start_token, 0), [batch_size, 1]), 
            axis=1
        )
        decoder_input = tf.concat([start_tokens, target_sequence[:, :-1, :]], axis=1)
        
        # Pass through LSTM layers
        x, _, _ = self.lstm1(decoder_input, initial_state=state1, training=training)
        x, _, _ = self.lstm2(x, initial_state=state2, training=training)
        
        # Generate output coordinates
        output = self.output_dense(x)
        
        return output
    
    def _autoregressive_forward(self, batch_size, state1, state2, training):
        """Autoregressive forward pass for inference."""
        outputs = []
        
        # Start with start token
        current_input = tf.expand_dims(
            tf.tile(tf.expand_dims(self.start_token, 0), [batch_size, 1]),
            axis=1
        )
        
        # Current LSTM states
        h1, c1 = state1
        h2, c2 = state2
        
        for step in range(self.max_sequence_length):
            # Single step through LSTM layers
            x, h1, c1 = self.lstm1(current_input, initial_state=[h1, c1], training=training)
            x, h2, c2 = self.lstm2(x, initial_state=[h2, c2], training=training)
            
            # Generate next coordinate
            next_coord = self.output_dense(x)
            outputs.append(next_coord)
            
            # Use output as next input
            current_input = next_coord
            
            # Check for early stopping (if all sequences in batch have ended)
            # This would require implementing end-of-sequence detection
            
        return tf.concat(outputs, axis=1)


class ShapeWeaverModel(keras.Model):
    """
    Complete ShapeWeaver model combining encoder and decoder.
    """
    
    def __init__(self, latent_dim=512, lstm_units=256, max_sequence_length=30000):
        """
        Initialize the complete model.
        
        Args:
            latent_dim (int): Dimensionality of the latent space
            lstm_units (int): Number of LSTM units in decoder
            max_sequence_length (int): Maximum curve sequence length
        """
        super(ShapeWeaverModel, self).__init__()
        
        self.encoder = ShapeWeaverEncoder(latent_dim)
        self.decoder = ShapeWeaverDecoder(latent_dim, lstm_units, max_sequence_length)
        
        self.latent_dim = latent_dim
        self.max_sequence_length = max_sequence_length
    
    def call(self, inputs, training=None):
        """
        Forward pass through the complete model.
        
        Args:
            inputs: Dictionary with 'image' and optionally 'target_sequence'
            training: Whether in training mode
            
        Returns:
            Generated curve sequence
        """
        image = inputs['image']
        target_sequence = inputs.get('target_sequence', None)
        
        # Encode image to context vector
        context_vector = self.encoder(image, training=training)
        
        # Decode context vector to curve sequence
        curve_sequence = self.decoder(
            context_vector, 
            target_sequence=target_sequence, 
            training=training
        )
        
        return curve_sequence
    
    def predict_curve(self, image):
        """
        Generate a curve for a single image.
        
        Args:
            image: Shape image [1, 256, 256, 1] or [256, 256, 1]
            
        Returns:
            Generated curve coordinates as numpy array
        """
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
        
        inputs = {'image': image}
        curve_sequence = self(inputs, training=False)
        
        return curve_sequence.numpy()
    
    def get_model_summary(self):
        """Get a summary of the model architecture."""
        # Build the model if not already built
        if not self.built:
            # Build with dummy input
            dummy_input = {
                'image': tf.zeros((1, 256, 256, 1)),
                'target_sequence': tf.zeros((1, 100, 2))
            }
            self(dummy_input, training=False)
        
        return {
            'encoder_params': self.encoder.count_params(),
            'decoder_params': self.decoder.count_params(),
            'total_params': self.count_params(),
            'latent_dim': self.latent_dim,
            'max_sequence_length': self.max_sequence_length
        }


def create_shapeweaver_model(latent_dim=512, lstm_units=256, max_sequence_length=30000):
    """
    Factory function to create a ShapeWeaver model.
    
    Args:
        latent_dim (int): Dimensionality of the latent space
        lstm_units (int): Number of LSTM units in decoder
        max_sequence_length (int): Maximum curve sequence length
        
    Returns:
        Compiled ShapeWeaver model
    """
    model = ShapeWeaverModel(
        latent_dim=latent_dim,
        lstm_units=lstm_units,
        max_sequence_length=max_sequence_length
    )
    
    return model


def test_model_architecture():
    """Test function to verify model architecture works correctly."""
    print("Testing ShapeWeaver model architecture...")
    
    # Create model
    model = create_shapeweaver_model(latent_dim=256, lstm_units=128, max_sequence_length=1000)
    
    # Test data
    batch_size = 2
    test_image = tf.random.normal([batch_size, 256, 256, 1])
    test_sequence = tf.random.uniform([batch_size, 100, 2])
    
    # Test encoder
    encoder_output = model.encoder(test_image)
    print(f"Encoder output shape: {encoder_output.shape}")
    
    # Test decoder with teacher forcing
    decoder_output = model.decoder(encoder_output, target_sequence=test_sequence, training=True)
    print(f"Decoder output shape (training): {decoder_output.shape}")
    
    # Test complete model
    inputs = {'image': test_image, 'target_sequence': test_sequence}
    model_output = model(inputs, training=True)
    print(f"Complete model output shape: {model_output.shape}")
    
    # Test inference mode (without target sequence)
    inference_inputs = {'image': test_image}
    inference_output = model(inference_inputs, training=False)
    print(f"Inference output shape: {inference_output.shape}")
    
    # Model summary
    summary = model.get_model_summary()
    print(f"Model summary: {summary}")
    
    print("Model architecture test completed successfully!")


if __name__ == "__main__":
    test_model_architecture()
