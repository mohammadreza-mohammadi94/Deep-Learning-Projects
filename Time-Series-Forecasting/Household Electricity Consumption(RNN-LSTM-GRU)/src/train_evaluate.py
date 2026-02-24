import numpy as np
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score)
from tensorflow.keras.callbacks import EarlyStopping
from src.utils import save_plot

def train_model(model, X_train, y_train, 
                X_test, y_test, epochs, batch_size, logger):
    """
    Train a given Keras model with early stopping.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled Keras model to be trained.
    X_train : np.ndarray
        Training input sequences.
    y_train : np.ndarray
        Training target values.
    X_test : np.ndarray
        Validation input sequences.
    y_test : np.ndarray
        Validation target values.
    epochs : int
        Number of training epochs.
    batch_size : int
        Number of samples per gradient update.
    logger : logging.Logger
        Logger instance for logging progress.
    """
    logger.info(f"Training model: {model.name}")
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping],
                        verbose=1)
    return history

def evaluate_and_plot(y_true, predictions, histories, 
                      target_scaler, model_names, logger):
    """
    Evaluate model predictions and generate comparison plots.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    predictions : list of np.ndarray
        List of model prediction arrays.
    histories : list of tf.keras.callbacks.History
        Training histories of the models.
    target_scaler : sklearn.preprocessing.MinMaxScaler
        Scaler used to inverse transform target values.
    model_names : list of str
        Names of the models (for labeling).
    logger : logging.Logger
        Logger instance for logging progress and results.
    """
    logger.info("Evaluating models and generating plots")
    
    y_true = np.expm1(target_scaler.inverse_transform(y_true.reshape(-1, 1))).reshape(y_true.shape)
    predictions = [np.expm1(target_scaler.inverse_transform(pred.reshape(-1, 1))).reshape(pred.shape) for pred in predictions]
    
    def print_metrics(y_true, y_pred, model_name):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        logger.info(f"{model_name} Metrics: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        print(f"{model_name} Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}\n")
    
    for pred, name in zip(predictions, model_names):
        print_metrics(y_true, pred, name)
    
    save_plot(
        [y_true[-100:].flatten()] + [pred[-100:].flatten() for pred in predictions],
        ['Actual'] + [f'{name} Prediction' for name in model_names],
        'Electricity Load Forecast: GRU vs CNN-GRU vs Transformer',
        'Time Step (Hours)',
        'Global Active Power (kW)',
        'electricity_predictions.png',
        ['-'] + ['--'] * len(predictions)
    )
    logger.info("Prediction plot saved as 'electricity_predictions.png'")
    
    loss_data = []
    loss_labels = []
    linestyles = []
    for history, name in zip(histories, model_names):
        loss_data.extend([history.history['loss'], history.history['val_loss']])
        loss_labels.extend([f'{name} Training Loss', f'{name} Validation Loss'])
        linestyles.extend(['-', '--'])
    
    save_plot(
        loss_data,
        loss_labels,
        'Training and Validation Loss',
        'Epoch',
        'MSE Loss',
        'electricity_loss_history.png',
        linestyles
    )
    logger.info("Loss history plot saved as 'electricity_loss_history.png'")
