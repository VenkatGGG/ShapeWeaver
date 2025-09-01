"""
ShapeWeaver Web Application

Flask-based web application for interactive shape drawing and curve generation.
Users can draw shapes on a canvas and watch AI-generated space-filling curves.
"""

import os
import sys
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import io
import base64

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.shapeweaver_model import create_shapeweaver_model


class ShapeWeaverApp:
    """Main application class for ShapeWeaver web interface."""
    
    def __init__(self, model_path=None):
        """
        Initialize the Flask application.
        
        Args:
            model_path (str): Path to trained model file
        """
        self.app = Flask(__name__, 
                        template_folder='../../templates',
                        static_folder='../../static')
        CORS(self.app)
        
        # Load trained model
        self.model = None
        self.model_path = model_path
        self.load_model()
        
        # Setup routes
        self.setup_routes()
    
    def load_model(self):
        """Load the trained ShapeWeaver model."""
        if self.model_path and os.path.exists(self.model_path):
            try:
                print(f"Loading model from {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating a new untrained model for demonstration...")
                self.model = create_shapeweaver_model(
                    latent_dim=64, 
                    lstm_units=32, 
                    max_sequence_length=5000
                )
        else:
            print("No model path provided. Creating untrained model for demonstration...")
            self.model = create_shapeweaver_model(
                latent_dim=64, 
                lstm_units=32, 
                max_sequence_length=5000
            )
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Serve the main application page."""
            return render_template('index.html')
        
        @self.app.route('/api/generate_curve', methods=['POST'])
        def generate_curve():
            """Generate a space-filling curve for uploaded shape."""
            try:
                data = request.get_json()
                
                if 'image_data' not in data:
                    return jsonify({'error': 'No image data provided'}), 400
                
                # Process the image data
                image_data = data['image_data']
                processed_image = self.process_image_data(image_data)
                
                if processed_image is None:
                    return jsonify({'error': 'Failed to process image'}), 400
                
                # Generate curve using the model
                curve_points = self.generate_curve_from_image(processed_image)
                
                # Return the generated curve
                return jsonify({
                    'success': True,
                    'curve_points': curve_points.tolist(),
                    'num_points': len(curve_points)
                })
                
            except Exception as e:
                print(f"Error in generate_curve: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/model_info')
        def model_info():
            """Get information about the loaded model."""
            if self.model:
                try:
                    # Build model if needed
                    if not self.model.built:
                        dummy_input = {
                            'image': tf.zeros((1, 256, 256, 1))
                        }
                        self.model(dummy_input, training=False)
                    
                    summary = self.model.get_model_summary()
                    
                    return jsonify({
                        'model_loaded': True,
                        'model_path': self.model_path,
                        'model_summary': summary
                    })
                except Exception as e:
                    return jsonify({
                        'model_loaded': True,
                        'model_path': self.model_path,
                        'error': f'Error getting model info: {e}'
                    })
            else:
                return jsonify({'model_loaded': False})
    
    def process_image_data(self, image_data):
        """
        Process image data from the frontend canvas.
        
        Args:
            image_data (str): Base64 encoded image data
            
        Returns:
            numpy.ndarray: Processed image array ready for model input
        """
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to grayscale and resize to 256x256
            image = image.convert('L')
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Apply threshold to create binary mask
            image_array = (image_array > 0.5).astype(np.float32)
            
            # Add channel dimension
            image_array = np.expand_dims(image_array, axis=-1)
            
            return image_array
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def generate_curve_from_image(self, image):
        """
        Generate curve points from processed image.
        
        Args:
            image (numpy.ndarray): Processed image array
            
        Returns:
            numpy.ndarray: Generated curve points [(x, y), ...]
        """
        try:
            # Add batch dimension
            image_batch = np.expand_dims(image, axis=0)
            
            # Generate curve using model
            inputs = {'image': image_batch}
            curve_sequence = self.model(inputs, training=False)
            
            # Remove batch dimension and convert to numpy
            curve_points = curve_sequence.numpy()[0]
            
            # Filter out padding (points that are likely repeated)
            valid_points = self.filter_valid_points(curve_points)
            
            return valid_points
            
        except Exception as e:
            print(f"Error generating curve: {e}")
            # Return a simple fallback curve
            return self.generate_fallback_curve()
    
    def filter_valid_points(self, curve_points, threshold=0.001):
        """
        Filter out padding and invalid points from generated curve.
        
        Args:
            curve_points (numpy.ndarray): Raw curve points from model
            threshold (float): Threshold for detecting repeated points
            
        Returns:
            numpy.ndarray: Filtered valid points
        """
        if len(curve_points) == 0:
            return curve_points
        
        valid_points = [curve_points[0]]
        
        for i in range(1, len(curve_points)):
            current_point = curve_points[i]
            previous_point = curve_points[i-1]
            
            # Check if point is significantly different from previous
            distance = np.sqrt(np.sum((current_point - previous_point) ** 2))
            
            if distance > threshold:
                valid_points.append(current_point)
            else:
                # If we hit repeated points, likely reached padding
                break
        
        return np.array(valid_points)
    
    def generate_fallback_curve(self):
        """Generate a simple fallback curve when model fails."""
        # Simple spiral pattern as fallback
        points = []
        center_x, center_y = 0.5, 0.5
        
        for i in range(100):
            t = i * 0.1
            radius = 0.3 * (1 - i / 100)
            x = center_x + radius * np.cos(t)
            y = center_y + radius * np.sin(t)
            points.append([x, y])
        
        return np.array(points)
    
    def run(self, host='localhost', port=5000, debug=True):
        """Run the Flask application."""
        print(f"Starting ShapeWeaver web application...")
        print(f"Open your browser to: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main function for running the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ShapeWeaver web application')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model file')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host to run the application on')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the application on')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Look for trained model if not specified
    if not args.model_path:
        possible_paths = [
            'logs/test_run/best_model.h5',
            'logs/test_run/final_model.h5',
            'models/shapeweaver_model.h5'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                args.model_path = path
                break
    
    # Create and run application
    app = ShapeWeaverApp(model_path=args.model_path)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
