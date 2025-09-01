# ShapeWeaver 🎨

An intelligent digital artist that transforms hand-drawn shapes into mesmerizing space-filling curves using deep learning.

## Overview

ShapeWeaver is a generative art project that sits at the intersection of computer vision, sequence modeling, and interactive design. Users draw simple closed shapes, and the AI automatically "weaves" continuous, non-intersecting lines that perfectly fill the interior of those shapes, visualized as captivating real-time animations.

## Project Phases

### Phase 1: Synthetic Dataset Generation
- Generate thousands of random polygon shapes as 256x256 binary masks
- Create corresponding space-filling curves using clipped Hilbert curves
- Build paired dataset of shape images and curve coordinates

### Phase 2: Model Architecture & Training
- **Encoder**: CNN that processes shape images into context vectors
- **Decoder**: LSTM that generates ordered sequences of curve points
- Train end-to-end on synthetic dataset

### Phase 3: Interactive Application
- Web-based drawing canvas for user input
- Real-time model inference
- Animated curve visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/VenkatGGG/ShapeWeaver.git
cd ShapeWeaver

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generate Dataset
```bash
python src/data_generation/generate_dataset.py --num_samples 10000
```

### Train Model
```bash
python src/training/train_model.py --epochs 100 --batch_size 32
```

### Run Web Application
```bash
python src/app/app.py
```

## Project Structure

```
ShapeWeaver/
├── src/
│   ├── data_generation/     # Phase 1: Dataset creation
│   ├── model/              # Phase 2: Neural network architecture
│   ├── training/           # Phase 2: Training pipeline
│   └── app/               # Phase 3: Web application
├── dataset/               # Generated training data
├── models/               # Saved trained models
├── static/              # Web app assets
└── templates/           # HTML templates
```

## Technical Details

- **Input**: 256x256 binary image masks of hand-drawn shapes
- **Output**: Ordered sequences of (x,y) coordinates forming space-filling curves
- **Architecture**: CNN Encoder + LSTM Decoder
- **Training**: Supervised learning on synthetic Hilbert curve data

## Contributing

This project is part of a generative art exploration. Feel free to contribute improvements, optimizations, or creative extensions!

## License

MIT License - See LICENSE file for details
