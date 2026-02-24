# Toxic Comment Classification with CNN + GloVe Embeddings

This project trains a 1D Convolutional Neural Network (CNN) to classify toxic comments into multiple categories using pre-trained GloVe word embeddings.

---

## 🧠 Task

Multi-label classification on text data to detect:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

---

## 🛠 Model Architecture

- **Embedding Layer** with GloVe vectors (100d)
- **3 x Conv1D + MaxPooling** layers
- **GlobalMaxPooling**
- **Dense + Dropout**
- **Sigmoid Output** for multi-label classification

---

## 📊 Dataset

- Jigsaw Toxic Comment Classification Challenge (`train.csv`)
- Loaded and preprocessed using Keras Tokenizer & Padding

---

## 📈 Evaluation

- Loss: `Binary Crossentropy`
- Metrics: `Accuracy`, `ROC-AUC per label`
- Visualized training loss and accuracy over epochs

---

## 🚀 How to Run

1. Download [GloVe embeddings](https://nlp.stanford.edu/projects/glove/)
2. Update paths to `glove.6B.100d.txt` and `train.csv`
3. Run:

```bash
python toxic_cnn.py
```


## 📦 Requirements

* TensorFlow / Keras
* Pandas, NumPy, Matplotlib
* scikit-learn
* GloVe (100d)

## 📊 Output

* Model summary
* Training/Validation plots
* ROC-AUC for each label

