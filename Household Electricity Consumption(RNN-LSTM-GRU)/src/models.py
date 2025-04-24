import tensorflow as tf

#-----------------#
# Creating Models #
#-----------------#
def weighted_mse(y_true, y_pred):
    """
    Compute the weighted mean squared error (MSE) between true and predicted values.

    This function calculates the MSE with different weights for each sample. Samples 
    where the true value is greater than the mean of `y_true` are given a higher weight 
    (2.0), while the rest are given a weight of 1.0. The MSE is then computed with these 
    weights, giving more importance to errors on samples with higher true values.
    """
    weights = tf.where(y_true > tf.reduce_mean(y_true), 2.0, 1.0)
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))

def rnn_model(seq_length, n_features):
    """
    Build and compile a Simple RNN model for time-series forecasting.

    Parameters
    ----------
    seq_length : int
        Number of past time steps in each input sequence.
    n_features : int
        Number of input features per time step.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_length, n_features)),
        tf.keras.layers.LSTM(128),  # حذف return_sequences=True
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(24)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=weighted_mse,
                  metrics=['mae'])
    return model

def lstm_model(seq_length, n_features):
    """
    Build and compile an LSTM model for time-series forecasting.

    Parameters
    ----------
    seq_length : int
        Number of past time steps in each input sequence.
    n_features : int
        Number of input features per time step.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_length, n_features)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),  # این لایه return_sequences ندارد
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(24)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss=weighted_mse, 
                  metrics=['mae'])
    return model

def gru_model(seq_length, n_features):
    """
    Build and compile a GRU model for time-series forecasting.

    Parameters
    ----------
    seq_length : int
        Number of past time steps in each input sequence.
    n_features : int
        Number of input features per time step.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_length, n_features)),
        tf.keras.layers.GRU(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(64),  # این لایه return_sequences ندارد
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(24)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss=weighted_mse, 
                  metrics=['mae'])
    return model