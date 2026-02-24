# ----------------
# Import Libraries
# ----------------
from __future__ import print_function, division
from builtins import range

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (Dense, 
                                     Input, 
                                     GlobalMaxPooling1D,
                                     Conv1D, 
                                     MaxPooling1D, 
                                     Embedding,
                                     Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import roc_auc_score


# ----------------
# Configuration
# ----------------
# Maximum length of text sequences (e.g., sentences or documents).
# Used to standardize input sizes by padding or truncating sequences to this length.
# Common in data preprocessing for RNNs or Transformers.
MAX_SEQUENCE_LENGTH = 100

# Maximum number of unique words to include in the model's vocabulary.
# Limits the vocabulary size to manage model complexity and memory usage.
# Used when tokenizing text data.
MAX_VOCAB_SIZE = 20000

# Dimensionality of the word embedding vectors.
# Each word is represented as a dense vector of this size.
# Used in the embedding layer to capture word meanings.
EMBEDDING_DIM = 100

# Fraction of the training data to be used as validation data.
# Helps evaluate the model during training and prevents overfitting.
# Used when splitting the dataset.
VALIDATION_SPLIT = 0.2

# Number of samples processed in each training batch.
# Affects the efficiency and stability of the training process.
# Used in the training loop of the model.
BATCH_SIZE = 128

# Number of times the model will iterate over the entire training dataset.
# Controls how long the model trains and how much it learns.
# Used to define the training duration.
EPOCHS = 10

# Load glove
print("\nLoading word vectors...")
word2vec = {}
with open(r"C:\Personal\Programming & Crypto\Courses\Deep Learning - Advanced NLP [Lazy Programmer]\Codes\files\glove.6B.100d.txt", "r", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype="float32")
        word2vec[word] = vec
print(f"Found {len(word2vec)} word vectors found.")


# ----------------
# Prepare Text Data and Labels
# ----------------
print("\n\nLoading comments...")
train = pd.read_csv(r"C:\Personal\Programming & Crypto\Courses\Deep Learning - Advanced NLP [Lazy Programmer]\Codes\datasets\Toxic Comment Classification\train.csv")
sentences = train["comment_text"].fillna("DUMMY_VALUE").values
possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult",
                   "identity_hate"]
targets = train[possible_labels].values

print(f"Max sequence length: {max(len(s) for s in sentences)}")
print(f"Min sequence length: {min(len(s) for s in sentences)}")
s = sorted(len(s) for s in sentences)
print(f"Median sequence length: {s[len(s) // 2]}")

# ----------------
# Tokenize and Pad Sequences
# ----------------
# Convert the sentences into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# Get word 
word2idx = tokenizer.word_index
print(f"Found {len(word2idx)} unique tokens.")

# Padding sequences 
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(f"Shape of data tensor: {data.shape}")

# ----------------
# Prepare Embedding Matrix
# ----------------
print("\n\nFilling pre-trained embeddings...")
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# ----------------
# Define Model Architecture
# ----------------
print("\nBuilding Model...")
# Input layer
inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), name="input_layer")

# Embedding layer with pre-trained GloVe weights
embedding = Embedding(
    input_dim=num_words,
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix],
    trainable=False,
    name="embedding_layer"
)(inputs)

# Convolutional layers with ReLU activation
x = Conv1D(filters=128, kernel_size=3, activation="relu", padding="same", name="conv1")(embedding)
x = MaxPooling1D(pool_size=3, name="pool1")(x)
x = Conv1D(filters=128, kernel_size=3, activation="relu", padding="same", name="conv2")(x)
x = MaxPooling1D(pool_size=3, name="pool2")(x)
x = Conv1D(filters=128, kernel_size=3, activation="relu", padding="same", name="conv3")(x)
x = GlobalMaxPooling1D(name="global_pool")(x)  # Reduce sequence dimension

# Dense layers with dropout for regularization
x = Dense(128, activation="relu", name="dense1")(x)
x = Dropout(0.3, name="dropout1")(x)  # Add dropout to prevent overfitting
outputs = Dense(len(possible_labels), activation="sigmoid", name="output_layer")(x)

# Create model
model = Model(inputs=inputs, outputs=outputs, name="Toxic_Comment_CNN")

# Compile model with RMSprop optimizer
model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(learning_rate=0.001),
    metrics=["accuracy"]
)

print("\n\n>>> MODEL SUMMARY <<<")
print(model.summary())

print("\n\nTraining the model...")
history = model.fit(
    data,
    targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
)

# ----------------
# Plot Training Results
# ----------------
# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ----------------
# Evaluate Model with ROC-AUC
# ----------------
print("\nEvaluating ROC-AUC...")
y_pred = model.predict(data)
auc_scores = [roc_auc_score(targets[:, i], y_pred[:, i]) for i in range(len(possible_labels))]
print(f"ROC-AUC scores for each label: {dict(zip(possible_labels, auc_scores))}")
print(f"Average ROC-AUC: {np.mean(auc_scores):.4f}")