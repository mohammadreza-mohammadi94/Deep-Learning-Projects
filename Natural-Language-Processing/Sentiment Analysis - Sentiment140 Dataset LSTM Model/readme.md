# Sentiment140 LSTM Sentiment Analysis

A simple LSTM-based sentiment classifier for tweets using the Sentiment140 dataset.

---

## Overview

This notebook/script:

1. Downloads and loads the Sentiment140 dataset (polarity: 0 = negative, 4 = positive).
2. Cleans tweets (removes URLs, mentions, hashtags, punctuation; lowercases text).
3. Samples 100,000 tweets and splits into train/test sets (80/20).
4. Tokenizes and pads sequences to a fixed length.
5. Defines a 2-layer LSTM model with dropout.
6. Trains the model with `ReduceLROnPlateau` and `ModelCheckpoint`.
7. Plots training & validation loss/accuracy curves.
8. Evaluates on the test set and shows sample predictions.

---

## Requirements

* Python 3.7+
* TensorFlow
* pandas, numpy, matplotlib, seaborn
* nltk, kagglehub

Install via:

```bash
pip install tensorflow pandas numpy matplotlib seaborn nltk kagglehub
```

---

## Usage

1. **Download dataset**
   The script uses `kagglehub` to fetch the `kazanova/sentiment140` dataset.

2. **Run preprocessing & training**

   ```bash
   python Sentiment140_LSTM.py
   ```

   (Or run the Colab notebook.)

3. **Outputs**

   * `app.log` — run logs
   * `sentiment140_metrics.png` — loss & accuracy plots
   * `sentiment140_model.h5` — best model checkpoint

---

## Notes

* **Label mapping bug**: Ensure you map `0 → 0`, `4 → 1` to preserve polarity.
* **Sample size**: Uses 100k tweets for faster iteration. Adjust `TRAIN_SAMPLES` for full dataset.
* **Tweaks**: Experiment with embedding dimension, LSTM units, epochs, and callbacks for better performance.
