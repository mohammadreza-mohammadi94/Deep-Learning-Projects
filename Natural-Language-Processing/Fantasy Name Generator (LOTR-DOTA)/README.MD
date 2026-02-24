# Fantasy Name Generator (LSTM)

This project builds a character-level LSTM neural network to generate fantasy-style names based on datasets from *Lord of the Rings* and *Dota*.

## 📌 Features

* Combines multiple fantasy name datasets
* Builds a character-level vocabulary
* Trains an LSTM model to predict the next character in a name
* Generates new names with or without a starting seed
* Uses temperature sampling to control creativity
* Saves the trained model

## 🚀 How It Works

1. **Load Data**
   Fetches name lists from online sources and cleans them.

2. **Preprocessing**
   Converts names into sequences of character indices, pads them, and prepares input/output pairs for training.

3. **Model Architecture**

   * Embedding layer
   * 2 LSTM layers (256 & 128 units)
   * Dropout layers for regularization
   * Dense output layer with softmax activation

4. **Training**
   Trains the model using `sparse_categorical_crossentropy` for 100 epochs.

5. **Name Generation**
   Predicts names character by character using temperature-controlled sampling.

## 📂 Datasets Used

* [LOTR Names](https://raw.githubusercontent.com/borhanMorphy/fantasy-name-generator/main/data/lotr_names.txt)
* [Dota Names](https://raw.githubusercontent.com/borhanMorphy/fantasy-name-generator/main/data/dota_names.txt)

## 📦 Requirements

```bash
pip install numpy pandas matplotlib tensorflow
```

## 🧪 How to Run

1. Run the full script in a Jupyter notebook or Colab environment.
2. The model will train and generate fantasy names like:

```
Randomly Generated Names:
Zarduin
Elroneth
Molduin
Tharien
Barathor
```

3. You can also generate names using custom seeds:

```
Seed: 'el' → Elgonar
Seed: 'gan' → Gandover
```

## 💾 Output

* Trained model saved as: `fantasy_name_generator_lotr_dota.h5`
* Logs written to: `logs.log`

## ✨ Example Use Cases

* Character name generation for games or stories
* Fantasy writing assistants
* Creative AI demos
