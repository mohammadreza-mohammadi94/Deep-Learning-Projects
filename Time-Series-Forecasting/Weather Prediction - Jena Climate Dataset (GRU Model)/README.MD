# 🌦️ Jena Climate Temperature Forecasting with GRU

This project builds a deep learning model using GRU (Gated Recurrent Unit) layers to predict hourly temperature based on the **Jena Climate Dataset (2009–2016)**. The dataset includes multiple weather-related features, and the model forecasts future temperatures by learning from historical sequences.

---

## 📁 Dataset

* **Source:** [Kaggle - Jena Climate](https://www.kaggle.com/datasets/mnassrib/jena-climate)
* **Period:** 2009 to 2016
* **Features Used:**

  * Temperature (T)
  * Pressure (p)
  * Relative Humidity (rh)
  * Wind Speed (wv)
  * Wind Direction (wd)

---

## 🧠 Model Overview

The model uses a univariate target (Temperature) and multivariate input features across a rolling time window to make predictions.

* **Architecture:**

  * GRU Layer (128 units)
  * Dense Layer (64 units, ReLU activation)
  * Output Layer (1 unit, linear)
* **Sequence Length:** 24 hours (1 day of hourly data)
* **Loss:** Mean Squared Error (MSE)
* **Optimizer:** Adam

---

## 🔄 Preprocessing Steps

* Hourly resampling and interpolation to fill missing values
* Time-based feature engineering:

  * Hour, Day of Week (encoded using sine/cosine functions)
* MinMax normalization of all input features
* Sliding window generation for sequences

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn kagglehub
```

### 2. Download the dataset

```python
import kagglehub
path = kagglehub.dataset_download("mnassrib/jena-climate")
```

### 3. Train the model

```python
predictor = TemperaturePredictor(data_path="/kaggle/input/jena-climate/jena_climate_2009_2016.csv")
X_train, y_train, X_test, y_test = predictor.load_and_preprocess_data()
predictor.build_model()
predictor.train(X_train, y_train)
```

### 4. Evaluate the model

```python
y_true, y_pred = predictor.evaluate(X_test, y_test)
predictor.plot_results(y_true, y_pred)
```

---

## 📊 Output

* **Metric:** Root Mean Squared Error (RMSE) on the test set
* **Plot:** True vs. Predicted Temperature Curve

---

## 📦 Files

* `jena_climate_2009_2016.csv` – Input dataset
* `Jena Climate.ipynb` – Full implementation in a Colab notebook
* `TemperaturePredictor` class – Encapsulates preprocessing, training, evaluation, and visualization

---

## 🛠️ Technologies Used

* Python 3.x
* TensorFlow / Keras
* Pandas, NumPy
* Matplotlib
* Scikit-learn
* KaggleHub (for dataset download)

---

## 📌 Future Improvements

* Add support for multistep forecasting
* Use Bidirectional or stacked GRU/LSTM layers
* Incorporate additional weather variables or external data sources
* Tune hyperparameters using tools like KerasTuner or Optuna

---

## 🤝 Acknowledgements

* Dataset from [Kaggle: Jena Climate Dataset](https://www.kaggle.com/datasets/mnassrib/jena-climate)
* Inspired by time series forecasting techniques using deep learning
