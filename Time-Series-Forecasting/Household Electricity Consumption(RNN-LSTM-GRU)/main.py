from src.preprocess import DataPreprocessor
from src.models import (rnn_model, 
                        lstm_model, 
                        gru_model)
from src.train_evaluate import train_model, evaluate_and_plot
from src.utils import setup_logging
from sklearn.model_selection import TimeSeriesSplit

def main():
    logger = setup_logging()
    SEQ_LENGTH = 12
    FORECAST_HORIZON = 24
    N_EPOCHS = 50
    BATCH_SIZE = 64
    
    preprocessor = DataPreprocessor('household_power_consumption.txt', logger)
    data = preprocessor.load_data()
    
    features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 
                'Hour', 'DayOfWeek', 'Month', 'Moving_Avg_24h', 'IsHoliday', 'IsPeak']
    target = 'Global_active_power'
    data_scaled, features, target = preprocessor.normalize_data(features, target)
    
    n_features = len([f for f in features if f != target])
    X, y = preprocessor.create_sequences(data_scaled, SEQ_LENGTH, FORECAST_HORIZON, n_features)
    
    tscv = TimeSeriesSplit(n_splits=5)
    histories = []
    predictions = []
    models = [
        ('GRU', gru_model(SEQ_LENGTH, n_features)),
        ('LSTM', lstm_model(SEQ_LENGTH, n_features)),
        ('RNN', rnn_model(SEQ_LENGTH, n_features))
    ]
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        fold_histories = []
        fold_predictions = []
        for name, model in models:
            history = train_model(model, X_train, y_train, X_test, y_test, N_EPOCHS, BATCH_SIZE, logger)
            fold_histories.append(history)
            pred = model.predict(X_test)
            fold_predictions.append(pred)
        
        evaluate_and_plot(y_test, fold_predictions, fold_histories, preprocessor.target_scaler, [name for name, _ in models], logger)
    
    logger.info("Execution completed successfully")

if __name__ == "__main__":
    main()

