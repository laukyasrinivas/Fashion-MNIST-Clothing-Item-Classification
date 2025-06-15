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

## Methodology

The following steps were followed to build and evaluate the clothing classification model:

1. **Dataset Loading & Preprocessing**
   - Loaded the Fashion MNIST dataset from `tf.keras.datasets`.
   - Normalized pixel values of the grayscale images (scale from 0–255 to 0–1).
   - Visualized a few sample images to understand the data distribution.

2. **Model Design**
   - Built a Convolutional Neural Network (CNN) using TensorFlow/Keras.
   - Architecture includes:
     - Two Conv2D + MaxPooling layers
     - One fully connected Dense layer with ReLU
     - Output layer with 10 neurons (logits for 10 clothing categories)

3. **Model Training**
   - Compiled the model using Adam optimizer and Sparse Categorical Crossentropy loss.
   - Trained the model for 10 epochs using the training set.

4. **Evaluation**
   - Evaluated model accuracy using the test dataset.
   - Plotted accuracy and loss across epochs to visualize learning progress.

5. **Prediction & Visualization**
   - Generated predictions on the test images.
   - Created visual plots to show:
     - Predicted vs actual labels
     - Confidence levels using probability bar charts

6. **Model Interpretation**
   - Highlighted correct vs incorrect predictions using color-coded labels.
   - Showcased grid visualizations for better interpretability of model behavior.

## Results

- Achieved around **91% test accuracy** after training for 10 epochs.
- Visualized predictions with confidence percentages.
- Plotted training accuracy and loss over epochs.
