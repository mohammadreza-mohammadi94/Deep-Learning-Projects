# Kuzushiji Character Recognition using LeNet5 and Custom AlexNet

This project implements Kuzushiji character recognition using two different deep learning architectures: LeNet5 and a custom AlexNet model. The dataset used is Kuzushiji-49, consisting of 49 classes of Japanese cursive characters, with images in 28x28 resolution.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Models](#models)
  - [LeNet5](#lenet5)
  - [Custom AlexNet](#custom-alexnet)
- [Results](#results)
- [Visualization](#visualization)
- [Usage](#usage)


## Dataset

The project uses the Kuzushiji-MNIST and Kuzushiji-49 datasets. You can download the datasets using the provided Python script, which retrieves files from the official [Kuzushiji-MNIST GitHub repository](https://github.com/rois-codh/kmnist).

- **Dataset Options:**
  - Kuzushiji-MNIST (10 classes, 70k examples)
  - Kuzushiji-49 (49 classes, 270k examples)

The dataset is loaded in NumPy format and normalized for training deep learning models.

## Installation

Make sure to have TensorFlow installed to run the models.

## Models

### LeNet5

LeNet5 is a classical convolutional neural network architecture designed for digit recognition. In this project, we adapted it to classify Kuzushiji characters.

**Architecture Overview:**
- Conv2D → AvgPool
- Conv2D → AvgPool
- Flatten → Dense(120) → Dense(84) → Output

Training the LeNet5 model achieves a performance of around 88% accuracy on the test dataset after 30 epochs.

### Custom AlexNet

A deeper AlexNet architecture was implemented to improve performance over LeNet5. The custom AlexNet model includes more convolutional layers, deeper dense layers, and dropout for regularization.

**Architecture Overview:**
- Conv2D → MaxPool
- Conv2D → MaxPool
- Conv2D → Conv2D → MaxPool
- Flatten → Dense(1024) → Dropout → Dense(512) → Dropout → Output

This model achieved about 94% accuracy on the test dataset after 10 epochs.

## Results

| Model       | Accuracy (Training) | Accuracy (Test) | Loss (Test) |
|-------------|---------------------|-----------------|-------------|
| LeNet5      | 88.24%               | 88.12%          | 0.5434      |
| Custom AlexNet | 94.02%             | 93.97%          | 0.2596      |

## Visualization

The model architectures can be visualized using `visualkeras`. Below is a visualization of the custom AlexNet model:

```python
import visualkeras
visualkeras.layered_view(model_alexnet)
```

This will generate a layered view of the custom AlexNet architecture.

## Usage

To run the project:
1. Download the dataset by running the `download_kmnist.py` script.
2. Train the models using the provided code:
   - LeNet5: `lenet5.py`
   - Custom AlexNet: `alexnet.py`
3. View training progress and accuracy plots after training.

