# Fashion-MNIST-Clothing-Item-Classification with CNN

This deep learning project focuses on the classification of clothing items using images from the Fashion MNIST dataset. A Convolutional Neural Network (CNN) is developed using TensorFlow and Keras to automatically learn and identify patterns in grayscale images of fashion products such as shirts, trousers, sneakers, bags, and more.

## What is Fashion MNIST?

Fashion MNIST is a benchmark dataset designed to replace the original MNIST (handwritten digits) as a more challenging computer vision task. It contains:

- **70,000 images**
- **Grayscale 28x28 images**
- **10 Clothing Categories**:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

## Project Objective

To design and train a Convolutional Neural Network that can:
- Accurately classify clothing items from unseen test images
- Visualize model performance using graphs and prediction confidence bars
- Demonstrate how CNNs learn hierarchical image features

## Tools & Technologies

- Python
- TensorFlow & Keras
- NumPy
- Matplotlib

## Model Architecture

The CNN used in this project includes:
- **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation  
- **MaxPooling2D Layer 1**: 2x2 pool size  
- **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation  
- **MaxPooling2D Layer 2**  
- **Flatten Layer**
- **Dense Layer**: 128 units, ReLU  
- **Output Layer**: 10 units (one for each class), raw logits

## Results

- Achieved around **91% test accuracy** after training for 10 epochs.
- Visualized predictions with confidence percentages.
- Plotted training accuracy and loss over epochs.
