# 🎥 YouTube Views Predictor

This project uses an LSTM neural network to predict the number of views a YouTube video might get based on its title, tags, category, likes, dislikes, and comments. The model is trained using real YouTube data from Kaggle.

## 📦 Features

* Downloads real YouTube video data via [KaggleHub](https://github.com/KaggleHub/kagglehub)
* Cleans and preprocesses video metadata
* Uses LSTM to learn patterns from textual and numerical features
* Predicts video view count
* Logs and visualizes training progress

## 🔧 Technologies Used

* Python
* TensorFlow / Keras
* LSTM Neural Networks
* KaggleHub
* Matplotlib, Seaborn
* Scikit-learn

## 🚀 How to Use

1. **Install dependencies** (in a virtual environment is recommended):

   ```bash
   pip install kagglehub pandas numpy seaborn matplotlib scikit-learn tensorflow
   ```

2. **Run the script**:

   ```bash
   python your_script_name.py
   ```

3. **Output**:

   * Trained LSTM model saved as `yt_views_model.h5`
   * Training metrics plotted in `training_history.png`
   * Predicted views for a sample input

## 📂 Dataset

* [YouTube Trending Videos Dataset](https://www.kaggle.com/datasnaek/youtube) (fetched using `kagglehub`)

## 🧠 Sample Prediction

The script includes a sample prediction:

```python
"Amazing AI tutorial video | ai, technology, tutorial"
```

## 📈 Output Visualization

Model loss and MAE curves are saved for inspection after training.
