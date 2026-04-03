# Age Prediction from Face Images

## Overview
This project is a baseline regression model built with **PyTorch** to predict a person's age based on their face image. It serves as an initial experiment to evaluate the performance of a standard Fully Connected Neural Network (Linear Layers) on image data before advancing to more complex architectures.

## Dataset
The dataset contains 2000 human face images with their corresponding ages. 
* **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/mohammad2012191/q2-ka-ai-2026)
* **Image Size:** 36x36 pixels (RGB)
* **Split:** 80% Training / 20% Testing

## Model Architecture
The baseline model is a 4-layer Fully Connected Neural Network (Multi-Layer Perceptron):
* **Input:** 3888 features (Flattened 3x36x36 image)
* **Hidden Layers:** 3 Linear layers with 128 units and ReLU activation.
* **Output:** 1 unit (Continuous value representing the predicted age).
* **Loss Function:** Mean Squared Error (MSE).
* **Optimizer:** AdamW.

## Results & Error Analysis
After training for 40 epochs, the model successfully learns the training data but struggles to generalize accurately on the test set. 

**Why does this happen?**
By using Linear Layers, the 2D images are flattened into a 1D vector. This process destroys the crucial **spatial relationships** and structural details of the face (e.g., the distance between eyes, wrinkles, etc.). 

**Future Work:** The next step is to replace the current baseline model with a **Convolutional Neural Network (CNN)**, which is specifically designed to extract and preserve spatial features from images, leading to much higher accuracy.

## How to Run
1. Clone this repository.
2. Install the required libraries: `pip install torch torchvision pandas numpy matplotlib scikit-learn kagglehub`
3. Run the Jupyter Notebook to download the data, train the model, and view the predictions.
