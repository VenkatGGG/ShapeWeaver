"""
ShapeWeaver Data Generation Module

This module contains utilities for generating synthetic training data
for the ShapeWeaver project, including shape generation and Hilbert
curve creation.
"""

from .shape_generator import ShapeGenerator
from .hilbert_curve import HilbertCurveGenerator
from .generate_dataset import DatasetGenerator

__all__ = ['ShapeGenerator', 'HilbertCurveGenerator', 'DatasetGenerator']
