# Heart Disease Prediction Using Deep Learning

This project involves building a deep learning model to predict heart disease based on patient data. It utilizes a neural network implemented with TensorFlow and Keras.

## Project Overview

1. **Data Loading and Exploration**
   - The dataset is loaded and explored to understand its structure and content. Key statistics and categorical data are examined.

2. **Data Preprocessing**
   - The features and target variable are separated.
   - Categorical features are encoded using Label Encoding.
   - Data is split into training and testing sets, and feature scaling is applied.

3. **Model Building**
   - A Sequential neural network model is created with multiple dense layers and dropout layers to prevent overfitting.
   - The model is compiled using the Adam optimizer and binary cross-entropy loss function.

4. **Training and Evaluation**
   - The model is trained with early stopping to prevent overfitting.
   - Training and validation accuracy and loss are plotted to evaluate performance.

## How to Run

1. **Install Dependencies:**

   Ensure you have the following libraries installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```

2. **Run the Notebook:**

   Open the notebook in a Jupyter environment or Google Colab and execute the cells.

## Files Included

- `Heart_Disease_Deep_Learning.ipynb`: The main Jupyter Notebook containing the code for data processing, model building, and evaluation.
