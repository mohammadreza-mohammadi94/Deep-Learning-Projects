# Electricity Consumption Forecasting with RNN, LSTM, and GRU

## Project Overview

This project focuses on forecasting electricity consumption (measured as `Global_active_power` in kilowatts) for a household over the next 24 hours using historical hourly data. Three deep learning models—Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU)—are implemented to predict future electricity usage based on past consumption patterns and temporal features such as hour of the day, day of the week, and month.

The project includes data preprocessing (handling missing values, resampling to hourly data, normalization), model training with early stopping, and evaluation using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score. Visualizations of predictions and training/validation loss are also generated to compare model performance.

## Dataset

The dataset used in this project is the **Household Electric Power Consumption Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption).

- **File**: `household_power_consumption.txt`
- **Description**:
  - Contains electricity consumption data for a single household from December 2006 to November 2010 (47 months).
  - Data is recorded at a minute-level granularity but is resampled to hourly data for this project.
- **Key Columns**:
  - `Date` and `Time`: Combined to create a `Datetime` column.
  - `Global_active_power`: Active power consumption in kilowatts (target variable for prediction).
  - Other columns like `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1`, `Sub_metering_2`, and `Sub_metering_3` are available but not used in this project.
- **Size**: Approximately 2 million rows (minute-level data), reduced to hourly data after preprocessing.

## Project Structure

The project is organized into modular Python scripts for better maintainability:

- `main.py`: Main script to run the project, orchestrating data preprocessing, model training, and evaluation.
- `preprocess.py`: Handles data loading, cleaning, resampling, feature extraction, and sequence creation.
- `models.py`: Defines the RNN, LSTM, and GRU models using TensorFlow/Keras.
- `train_evaluate.py`: Contains functions for training models and evaluating their performance, including visualization.
- `utils.py`: Utility functions for logging setup and plot saving.
- `household_power_consumption.txt`: The dataset file (must be downloaded and placed in the project directory).
- `electricity_forecast.log`: Log file generated during execution.
- `electricity_predictions.png`: Plot comparing actual vs. predicted electricity consumption.
- `electricity_loss_history.png`: Plot showing training and validation loss for all models.

## Requirements

To run this project, you need the following dependencies:

- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - tensorflow

Install the dependencies using:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

## Methodology

### 1. **Data Preprocessing**
- **Loading and Cleaning**:
  - Load the dataset and combine `Date` and `Time` into a `Datetime` column.
  - Convert numerical columns to `float`, replacing invalid values (`?`) with `NaN` and filling missing values with column means.
- **Resampling**:
  - Resample minute-level data to hourly data by taking the mean of each hour.
- **Feature Engineering**:
  - Extract temporal features: `Hour`, `DayOfWeek`, and `Month` from the `Datetime` column.
  - Features used for prediction: `Global_active_power` (past values), `Hour`, `DayOfWeek`, and `Month`.
- **Normalization**:
  - Normalize features and target using `MinMaxScaler` to scale values to the range `[0, 1]`.
- **Sequence Creation**:
  - Create sequences with a length of 168 hours (1 week) to predict the next 24 hours of `Global_active_power`.

### 2. **Model Architecture**
Three models are implemented:
- **RNN Model**:
  - Input layer: `(seq_length=168, n_features=3)`
  - Two SimpleRNN layers (128 and 64 units) with dropout (0.2).
  - Dense output layer with 24 units (24-hour forecast).
- **LSTM Model**:
  - Input layer: `(seq_length=168, n_features=3)`
  - Two LSTM layers (128 and 64 units) with dropout (0.2).
  - Dense output layer with 24 units.
- **GRU Model**:
  - Input layer: `(seq_length=168, n_features=3)`
  - Two GRU layers (128 and 64 units) with dropout (0.2).
  - Dense output layer with 24 units.
- **Training Configuration**:
  - Loss function: Mean Squared Error (MSE).
  - Optimizer: Adam with a learning rate of 0.001.
  - Epochs: 30.
  - Batch size: 64.
  - Early stopping with patience of 5 to prevent overfitting.

### 3. **Training and Evaluation**
- **Train-Test Split**:
  - 80% training, 20% testing.
- **Training**:
  - Train each model (RNN, LSTM, GRU) on the training data with early stopping.
- **Evaluation**:
  - Evaluate models using MSE, MAE, and R² metrics.
  - Generate plots:
    - `electricity_predictions.png`: Compares actual vs. predicted values for the last 100 hours.
    - `electricity_loss_history.png`: Shows training and validation loss curves for all models.

## How to Run

1. **Download the Dataset**:
   - Download the dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption).
   - Place the `household_power_consumption.txt` file in the project directory.

2. **Set Up the Environment**:
   - Create a virtual environment (optional but recommended):
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install the required dependencies:
     ```bash
     pip install pandas numpy matplotlib scikit-learn tensorflow
     ```

3. **Run the Project**:
   - Ensure all scripts (`main.py`, `preprocess.py`, `models.py`, `train_evaluate.py`, `utils.py`) and the dataset are in the same directory.
   - Run the main script:
     ```bash
     python main.py
     ```

4. **View Results**:
   - **Logs**: Check `electricity_forecast.log` for detailed execution logs.
   - **Metrics**: Model performance metrics (MSE, MAE, R²) will be printed to the console.
   - **Plots**:
     - `electricity_predictions.png`: Comparison of actual vs. predicted electricity consumption.
     - `electricity_loss_history.png`: Training and validation loss curves for all models.

## Expected Outputs

- **Log File**: `electricity_forecast.log` contains detailed logs of the execution process.
- **Metrics**:
  - MSE, MAE, and R² scores for each model (RNN, LSTM, GRU) will be printed to the console.
- **Plots**:
  - `electricity_predictions.png`: Visualizes the actual vs. predicted electricity consumption for the last 100 hours.
  - `electricity_loss_history.png`: Displays the training and validation loss curves for all models over the epochs.

## Future Improvements

- **Additional Features**:
  - Incorporate other columns like `Voltage`, `Global_reactive_power`, or weather data to improve prediction accuracy.
- **Hyperparameter Tuning**:
  - Use grid search or random search to optimize hyperparameters like learning rate, number of units, and dropout rate.
- **Advanced Models**:
  - Experiment with hybrid models like CNN-LSTM or Transformer-based architectures.
- **Cross-Validation**:
  - Implement time-series cross-validation using `TimeSeriesSplit` to better validate the models.
- **Longer Forecasts**:
  - Extend the forecast horizon to predict electricity consumption for a week or a month.

## Troubleshooting

- **TensorFlow CPU Warning**:
  - If you see a warning about CPU instructions (e.g., `SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA`), it does not affect the functionality but may impact performance. You can ignore it or install a CPU-optimized version of TensorFlow (`pip install tensorflow-cpu`).
- **File Not Found**:
  - Ensure the dataset file (`household_power_consumption.txt`) is in the project directory.
- **Memory Issues**:
  - The dataset is large (~2 million rows). If you encounter memory issues, consider reducing the dataset size (e.g., use data from a specific year) or running on a machine with more RAM.
---

*Last updated: April 23, 2025*
