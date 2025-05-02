# 📰 Fake News Classification with LSTM (TensorFlow/Keras)

This project implements a **Fake News Detection** system using **Natural Language Processing (NLP)** and a **Bidirectional LSTM (Long Short-Term Memory)** network built with **TensorFlow/Keras**. The model is trained on a labeled dataset of news articles to classify whether a given article is **real** or **fake**.

---

## 📂 Dataset

The dataset used is from the **[Fake News Detection competition on Kaggle](https://www.kaggle.com/competitions/fake-news)**. It contains news article metadata and content labeled as real or fake.

* `train.csv`: Used to train the model (`text`, `title`, `label`)
* `test.csv`: Unlabeled data used for prediction after model training

---

## 🧠 Model Overview

This project uses a **Bidirectional LSTM** neural network for sequence modeling:

* **Embedding Layer** (to learn word representations)
* **Bidirectional LSTM** (to capture contextual dependencies)
* **Fully Connected Layers** with **Dropout** and **L2 Regularization**
* **Sigmoid Output** for binary classification (Fake = 1, Real = 0)

---

## 🔧 Features

* Custom **text preprocessing pipeline** (tokenization, lemmatization, stopword removal)
* **Word clouds** and **label distribution plots** for EDA
* **Tokenizer and padding** for LSTM input preparation
* **Model checkpointing** and **early stopping** for robust training
* **Evaluation metrics**: Accuracy, Confusion Matrix, Classification Report, ROC AUC

---

## 🛠️ Dependencies

```bash
tensorflow
nltk
pandas
numpy
matplotlib
seaborn
scikit-learn
wordcloud
pickle
```

---

## 🚀 How to Run

1. **Setup Kaggle API**:

   * Place your `kaggle.json` API key in the working directory.
   * The script moves and sets permissions automatically.

2. **Download the dataset**:

   ```bash
   kaggle competitions download -c fake-news
   unzip fake-news.zip
   ```

3. **Run the notebook or script**:

   * Includes EDA, training, evaluation, and final prediction steps.

---

## 📦 Output

* `best_model.h5.keras`: Saved best model checkpoint
* `fake_news_lstm_model.pkl`: Serialized model using `pickle`
* `test.csv` with a new `predictions` column added

---

## 📊 Visualizations

* Word clouds for Fake and Real news
* Label distribution bar plot
* ROC Curve for model evaluation

---

## 🧪 Evaluation Metrics

* **Accuracy**
* **Confusion Matrix**
* **Classification Report** (Precision, Recall, F1)
* **ROC AUC Curve**

---

## 📌 Project Structure

```
├── Fake News Classification - LSTM Model.ipynb
├── train.csv
├── test.csv
├── best_model.h5.keras
├── fake_news_lstm_model.pkl
├── README.md
└── kaggle.json
```

---

## 📬 Contact

For any questions, feel free to reach out or open an issue.
