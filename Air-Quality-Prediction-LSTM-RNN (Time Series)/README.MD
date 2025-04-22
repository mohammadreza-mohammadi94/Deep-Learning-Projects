# Air Quality Prediction with RNN and LSTM

## Project Overview

This project aims to predict the Air Quality Index (AQI) for the city of Delhi using time-series data from the "Air Quality Data in India" dataset. Two deep learning models, Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM), are implemented to forecast AQI based on historical air quality features such as PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, and Xylene.

The project includes data preprocessing steps (log transformation, handling missing values, and normalization), model training with learning rate scheduling and early stopping, and evaluation with metrics like MSE, MAE, and R². Visualizations of predictions and training/validation loss are also generated.

## Dataset

The dataset used in this project is sourced from Kaggle: Air Quality Data in India. It contains hourly air quality measurements for various cities in India. For this project, we focus on data for Delhi.

- **File**: `city_hour.csv`
- **Features**: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Target**: AQI (Air Quality Index)

## Requirements

To run this project, you need the following dependencies:

- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - tensorflow
  - kagglehub (for downloading the dataset)

You can install the dependencies using:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow kagglehub
```

## Project Structure

- `Air-Quality-LSTM-RNN.ipynb`: Main Jupyter notebook containing the code for data loading, preprocessing, model training, and evaluation.
- `city_hour.csv`: Dataset file (downloaded via kagglehub).
- `air_quality.log`: Log file containing debug and info messages generated during execution.
- `distribution.png`: Histogram plot of feature distributions.
- `aqi_predictions.png`: Plot comparing actual AQI with RNN and LSTM predictions.
- `aqi_loss_history.png`: Plot showing training and validation loss over epochs.

## Methodology

1. **Data Preprocessing**:

   - Filter data for Delhi.
   - Convert the `Datetime` column to a datetime format.
   - Apply log transformation (`np.log1p`) to the AQI target to handle skewness (features are not log-transformed).
   - Fill missing values with the mean of each column.
   - Normalize the data using `RobustScaler` to handle outliers effectively.

2. **Sequence Creation**:

   - Create sequences of length 48 (hours) for time-series modeling.
   - Features are used to predict the AQI at the next time step.

3. **Model Architecture**:

   - **RNN Model**:
     - Two SimpleRNN layers (128 and 64 units) with dropout (0.3).
     - Dense layers (32 units with ReLU activation, and 1 output unit).
   - **LSTM Model**:
     - Two LSTM layers (128 and 64 units) with dropout (0.3).
     - Dense layers (32 units with ReLU activation, and 1 output unit).
   - Both models use Huber Loss (robust to outliers) and Adam optimizer with an initial learning rate of 0.0001.

4. **Training**:

   - Train-test split: 80% training, 20% testing.
   - Train for 50 epochs with a batch size of 32.
   - Use `ReduceLROnPlateau` (factor=0.5, patience=3) to adjust the learning rate dynamically.
   - Use `EarlyStopping` (patience=20) to stop training if validation loss does not improve.

5. **Evaluation**:

   - Evaluate models using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.
   - Generate plots for predictions and training/validation loss.

## Latest Results

The models were trained and evaluated on April 22, 2025. Below are the latest performance metrics:

- **RNN Metrics**:

  - MSE: 257.0640
  - MAE: 11.3550
  - R²: 0.9806

- **LSTM Metrics**:

  - MSE: 239.3246
  - MAE: 10.5332
  - R²: 0.9819

### Analysis

- Both models show strong performance with high R² scores (&gt;0.98), indicating they explain most of the variance in AQI.
- LSTM slightly outperforms RNN with a lower MSE (239.32 vs. 257.06) and MAE (10.53 vs. 11.35).
- The models are effective at capturing general trends in AQI, but further improvements may be needed to better predict sudden peaks (e.g., high AQI values).

### Generated Plots

- `aqi_predictions.png`: Comparison of actual AQI with RNN and LSTM predictions over time.
- `aqi_loss_history.png`: Training and validation loss curves for both models.

## How to Run

1. **Clone the Repository** (if applicable):

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Download the Dataset**: The dataset is automatically downloaded using `kagglehub`. Ensure you have the Kaggle API set up with your credentials.

3. **Run the Code**:

   - Open the notebook (`Air-Quality-LSTM-RNN.ipynb`) in Jupyter Notebook or Google Colab.
   - Alternatively, run the Python script directly:

     ```bash
     python air_quality_prediction.py
     ```

4. **View Results**:

   - Check the console output for metrics.
   - View the generated plots (`aqi_predictions.png` and `aqi_loss_history.png`).
   - Review the log file (`air_quality.log`) for detailed execution logs.

## Future Improvements

- **Time-Series Cross-Validation**: Use `TimeSeriesSplit` to better validate the model on time-series data and prevent data leakage.
- **Feature Engineering**: Explore additional features (e.g., weather data) to improve prediction accuracy.
- **Advanced Models**: Experiment with Transformer-based models or hybrid CNN-LSTM architectures for better performance.
- **Hyperparameter Tuning**: Use grid search or random search to optimize hyperparameters like learning rate, batch size, and network architecture.
- **Peak Prediction**: Focus on improving predictions for sudden AQI spikes by using weighted loss functions or data augmentation.
