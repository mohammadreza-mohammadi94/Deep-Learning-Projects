# CIFAR-10 Image Classification with ResNet

This project implements a Convolutional Neural Network (CNN) based on the ResNet architecture to classify images from the CIFAR-10 dataset. The model was trained and evaluated using **PyTorch**, achieving an accuracy of approximately **81%** on the test set.

## Project Overview

The primary objective of this project is to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes. The project leverages a customized ResNet model for this purpose.

## Model Architecture

The model is based on the **ResNet-18** architecture, utilizing a series of convolutional layers with batch normalization and residual connections to improve training efficiency and overall performance.

## Key Features

- **Data Augmentation**: Basic normalization has been applied to the CIFAR-10 dataset.
- **Model Architecture**: ResNet-18 is used for image classification.
- **Evaluation Metrics**: Accuracy is used to evaluate model performance on the training, validation, and test sets.
- **Confusion Matrix**: A confusion matrix is generated to visualize classification performance across the 10 classes.


## Results

- **Training Accuracy**: Varies by epoch.
- **Validation Accuracy**: Varies by epoch.
- **Test Accuracy**: Approximately **81%**.

## Confusion Matrix

The confusion matrix provides insight into the model's performance across different classes in the CIFAR-10 dataset.

## Technologies Used

- **PyTorch** for model building and training.
- **Torchvision** for dataset handling and data transformations.
- **Matplotlib** for visualizing the confusion matrix and results.
