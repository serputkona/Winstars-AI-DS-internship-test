# MNIST Classifier Project

This repository contains implementations of three different machine learning models for classifying handwritten digits from the MNIST dataset. The project includes Random Forest, Neural Network, and Convolutional Neural Network implementations.

## Overview

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). This project provides a unified interface for training and making predictions with different classifier types:

- **Random Forest Classifier**: A traditional machine learning approach
- **Neural Network**: A feed-forward neural network with dense layers
- **Convolutional Neural Network**: A deep learning approach designed specifically for image data

## Project Structure

The project uses an abstract base class that defines a common interface for all classifier implementations:

```
MnistClassifierInterface (Abstract Base Class)
├── MnistRFClassifier
├── MnistNNClassifier
└── MnistCNNClassifier
```

All implementations are wrapped in a single `MnistClassifier` class that provides a unified entry point.

## Setup and Installation

### Prerequisites

This project requires the following dependencies:

- Python 3.6+
- TensorFlow 2.x
- Keras
- scikit-learn
- NumPy
- Matplotlib
- Seaborn

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/mnist-classifier.git
   cd mnist-classifier
   ```

2. Install the required packages:
   ```
   pip install tensorflow numpy scikit-learn matplotlib seaborn
   ```

## Usage

### Basic Usage

```python
import numpy as np

# Assume the classifier is already created and trained
classifier = MnistClassifier('rf')
classifier.train()  # Train the model if it's not trained yet

# Create a random 28x28 image (simulating a real digit image)
sample_image = np.random.rand(28, 28) * 255  # Simulating a digit image

# Add batch dimension (1, 28, 28) to match the expected input format
sample_image = np.expand_dims(sample_image, axis=0)

# Predict the digit class but need to be replaced with actual data
prediction = classifier.predict(sample_image)

```

### Choosing an Algorithm

The `MnistClassifier` class accepts one of three algorithm options:

- `'rf'`: Random Forest (traditional machine learning)
- `'nn'`: Neural Network (feed-forward network)
- `'cnn'`: Convolutional Neural Network (deep learning)

Example:
```python
# For Random Forest
rf_classifier = MnistClassifier(algorithm='rf')

# For Neural Network
nn_classifier = MnistClassifier(algorithm='nn')

# For CNN
cnn_classifier = MnistClassifier(algorithm='cnn')
```

## Model Details

### Random Forest Classifier

- Uses scikit-learn's RandomForestClassifier
- Optimized hyperparameters:
  - 200 estimators (decision trees)
  - min_samples_leaf=1
  - min_samples_split=2
- Features flattened 28x28 images (784 features)

### Neural Network

- Feed-forward architecture with:
  - Dense layers (256 → 128 → 64 → 10 neurons)
  - ReLU activation (softmax for output)
  - Dropout and L2 regularization to prevent overfitting
- Data augmentation for selected classes
- Learning rate scheduling

### Convolutional Neural Network

- Architecture:
  - Two convolutional blocks with batch normalization
  - Each block: Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPooling → Dropout
  - Fully connected layers at the end
- Uses categorical cross-entropy loss
- Adam optimizer

## Evaluation

All models include comprehensive evaluation metrics:

- Classification accuracy
- Detailed classification reports
- Confusion matrices
- Analysis of misclassified samples
- Class-specific error analysis

## Performance Comparison

Below is a general comparison of the models:

| Model | Advantages | Disadvantages |
|-------|------------|---------------|
| Random Forest | Fast training, interpretable | Less accurate for complex patterns |
| Neural Network | Good balance of performance and speed | Requires more hyperparameter tuning |
| CNN | Best accuracy for image data | Slowest training time |

## Example Outputs

During training, each model will display:
- Training progress
- Test accuracy
- Classification report
- Misclassification statistics
- Confusion matrix (for NN and CNN)

When making predictions, the model will output the predicted digit labels.

