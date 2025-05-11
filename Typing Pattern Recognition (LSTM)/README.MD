# Typing Pattern Recognition using RNN (LSTM)

This project builds a Recurrent Neural Network (RNN) with LSTM layers to recognize and predict typing patterns based on keystroke dynamics. It uses timing features such as key hold time and key intervals from a dataset of user typing behavior.

## 📁 Dataset

* **Source**: [DSL-StrongPasswordData.csv](https://www.cs.cmu.edu/~keystroke/)
* **Features Used**:

  * `H.` (Key Hold Time)
  * `UD.` (Key Up-Down Time)
  * `DD.` (Key Down-Down Time)
* **Label**: No explicit classification label. This is a sequence regression task to predict the next timing pattern.

## 🧠 Model Architecture

* Input: Sequence of timing vectors
* Layers:

  * LSTM (64 units)
  * Dense (32 units, ReLU)
  * Dense (output\_dim, Linear)
* Output: Next time step's feature vector (regression)

## 🚀 How It Works

1. **Load and Normalize Data**
   Loads keystroke features and normalizes them between 0 and 1.

2. **Sequence Generation**
   Converts user keystroke logs into overlapping sequences suitable for RNN input.

3. **Model Training**
   The model learns to predict the next feature vector based on a sequence of previous keystrokes.

4. **Prediction**
   Given a sample typing sequence, the model predicts the next step's pattern.

## 🔧 Dependencies

* Python 3.x
* NumPy
* Pandas
* Matplotlib
* Seaborn
* TensorFlow / Keras

Install the required packages with:

```bash
pip install numpy pandas matplotlib seaborn tensorflow
```

## 🛠️ How to Run

1. Place `DSL-StrongPasswordData.csv` in the working directory or update the `file_path` in the code.
2. Run the script or Jupyter Notebook.
3. The model will be trained and saved as `keystroke_pattern_model.h5`.

## 📈 Example Output

```
Number of Users: 51
Number of Features: 31
Number of Sequences: 10294
Predicted Pattern: [0.1034 0.0865 ... 0.1221]
```

## 🧾 Logs

Logs are saved to `app_logs.log` and also streamed to console using Python's `logging` module.

## 📦 Output

* Trained Model: `keystroke_pattern_model.h5`

## 🔒 Applications

* User authentication
* Behavioral biometrics
* Typing pattern analytics
