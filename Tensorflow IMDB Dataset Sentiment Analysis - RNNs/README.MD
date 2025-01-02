# Sentiment Analysis using RNN, LSTM, GRU, and Bidirectional LSTM

This project demonstrates sentiment analysis using multiple recurrent neural network (RNN) architectures, including SimpleRNN, GRU, LSTM, and Bidirectional LSTM, on the IMDB movie reviews dataset.

## 📂 **Project Structure**
- **Data:** Uses the IMDB dataset from Keras, which is a collection of 25,000 movie reviews labeled as positive or negative.
- **Models:**
  - **Simple RNN**
  - **GRU (Gated Recurrent Unit)**
  - **LSTM (Long Short-Term Memory)**
  - **Bidirectional LSTM**
  
## 🛠️ **Requirements**
- TensorFlow
- Keras
- NumPy
- Matplotlib

## 📝 **Steps to Run**
1. **Load and Preprocess Data:**
   - The dataset is loaded using `imdb.load_data()`.
   - Text sequences are padded to a uniform length of 400.
   
2. **Define Models:**
   - **RNN Model:** A simple RNN with 128 units and a Dense layer with a sigmoid activation.
   - **GRU Model:** A GRU-based model with 128 units and a sigmoid output.
   - **LSTM Model:** An LSTM-based model with 128 units and a sigmoid output.
   - **Bidirectional LSTM Model:** Bidirectional LSTM with 128 units and a sigmoid output.

3. **Compile and Train:**
   - Each model is compiled using the `adam` optimizer and `binary_crossentropy` loss function.
   - Models are trained for 5 epochs with a batch size of 64.

4. **Evaluation:**
   - Each model is evaluated on the test dataset to calculate accuracy.

## 📊 **Results**
- The models are evaluated based on accuracy on the test dataset.
- Training and validation loss for each model are displayed during training.
