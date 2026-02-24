import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from src.utils import save_plot

def train_model(model, X_train, y_train, X_test, y_test,
                epochs, batch_size, logger):
    """
    Train the model with early stopping.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled Keras model to be trained.
    X_train : np.ndarray
        Training input data.
    y_train : np.ndarray
        Training target values.
    X_test : np.ndarray
        Validation input data.
    y_test : np.ndarray
        Validation target values.
    epochs : int
        Number of training epochs.
    batch_size : int
        Number of samples per gradient update.
    logger : logging.Logger
        Logger instance for logging progress.
    """
    logger.info(f"Traning model : {model.name}")
    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    # Train the model
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping],
                        verbose=1)
    return history


def evaluate_and_plot(models, histories, X_test, y_test, class_names, logger):
    """
    Evaluate the models and generate plots.

    Parameters
    ----------
    models : list of tf.keras.Model
        List of trained Keras models.
    histories : list of tf.keras.callbacks.History
        Training histories of the models.
    X_test : np.ndarray
        Test input data.
    y_test : np.ndarray
        Test target values.
    class_names : list of str
        Names of the classes for labeling.
    logger : logging.Logger
        Logger instance for logging progress and results.
    """
    logger.info("Evaluating models and generating plots")

    # Predictions
    predictions = []
    for model in models:
        y_pred_probs = model.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        predictions.append(y_pred)

    # Compute accuracy metrics
    for model, y_pred in zip(models, predictions):
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"{model.name} Accuracy: {accuracy:.4f}")
        print(f"{model.name} Accuracy: {accuracy:.4f}\n")
        print(f"{model.name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix: {model.name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'sentiment_confusion_matrix_{model.name}.png')
        plt.close()
        logger.info(f"Confusion matrix for {model.name} saved as 'sentiment_confusion_matrix_{model.name}.png'")

    # loss plot
    loss_data = []
    loss_labels = []
    linestyles = []
    for history, model in zip(histories, models):
        loss_data.extend([history.history['loss'], history.history['val_loss']])
        loss_labels.extend([f'{model.name} Training Loss', f'{model.name} Validation Loss'])
        linestyles.extend(['-', '--'])
    
    save_plot(
        loss_data,
        loss_labels,
        'Training and Validation Loss',
        'Epoch',
        'Loss',
        'sentiment_loss_history.png',
        linestyles
    )
    logger.info("Loss history plot saved as 'sentiment_loss_history.png'")

    # accuracy plot
    accuracy_data = []
    accuracy_labels = []
    for history, model in zip(histories, models):
        accuracy_data.extend([history.history['accuracy'], history.history['val_accuracy']])
        accuracy_labels.extend([f'{model.name} Training Accuracy', f'{model.name} Validation Accuracy'])
    
    save_plot(
        accuracy_data,
        accuracy_labels,
        'Training and Validation Accuracy',
        'Epoch',
        'Accuracy',
        'sentiment_accuracy_history.png',
        linestyles
    )
    logger.info("Accuracy history plot saved as 'sentiment_accuracy_history.png'")