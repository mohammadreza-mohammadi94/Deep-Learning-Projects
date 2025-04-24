# FashionMNIST-LeNet5

This repository implements the LeNet-5 architecture for image classification on the FashionMNIST dataset using TensorFlow and Keras. The project covers the essential steps for building, training, and evaluating a convolutional neural network.

## Project Structure:
1. **Import Libraries**: Import necessary libraries such as TensorFlow and Keras.
2. **Import FashionMNIST Dataset**: Load and explore the FashionMNIST dataset.
3. **Check Samples**: Visualize sample images from the dataset with corresponding labels.
4. **Normalization**: Normalize pixel values and reshape the images to fit the model's input.
5. **Model Definition**: Define the LeNet-5 architecture using Keras and TensorFlow.
   - 5.1 **Model Training**: Train the model using 45 epochs and batch size of 128.
   - 5.2 **Model Performance**: Evaluate and visualize the model’s performance metrics (accuracy and loss).

## Model Performance:
- **Accuracy**: 87.79%
- **Loss**: 0.345

## How to Run:
1. Clone the repository.
2. Install the necessary dependencies (`tensorflow`, `numpy`, `matplotlib`).
3. Run the provided code to train the model and evaluate its performance.

## References:
- Dataset: [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
- Model: LeNet-5
