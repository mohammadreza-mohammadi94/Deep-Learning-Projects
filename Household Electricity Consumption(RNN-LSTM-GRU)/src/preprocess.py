import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

class DataPreprocessor:
    def __init__(self, file_path, logger):
        self.file_path = file_path
        self.logger = logger
        self.data = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def remove_outliers(self, df, column):
        """
        Remove outliers from a specified column in a DataFrame using the IQR method.

        This method identifies outliers as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR,
        where Q1 and Q3 are the 25th and 75th percentiles respectively. Instead of removing
        the rows, the outlier values are clipped to the lower or upper bounds.
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower_bound, upper_bound)
        return df

    def load_data(self):
        """
        Load, clean, and preprocess the dataset.

        Reads the dataset from the specified CSV file path, handles missing values,
        combines date and time into a datetime column, resamples data to hourly intervals,
        and extracts time-based features for modeling.
        
        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame with hourly-aggregated values and additional time features.
        """
        self.logger.info("Loading and preprocessing data")
        try:
            df = pd.read_csv(self.file_path, sep=';', low_memory=False)
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
            df = df.drop(['Date', 'Time'], axis=1)
            
            for col in ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 
                        'Sub_metering_1', 'Sub_metering_2']:
                df[col] = df[col].replace('?', np.nan).astype(float)
            
            for col in df.columns:
                if col != 'Datetime':
                    df[col] = df[col].fillna(df[col].mean())
            
            df.set_index('Datetime', inplace=True)
            df_hourly = df.resample('H').mean().reset_index()
            
            df_hourly['Hour'] = df_hourly['Datetime'].dt.hour
            df_hourly['DayOfWeek'] = df_hourly['Datetime'].dt.dayofweek
            df_hourly['Month'] = df_hourly['Datetime'].dt.month
            df_hourly['Moving_Avg_24h'] = df_hourly['Global_active_power'].rolling(window=24).mean().fillna(method='bfill')
            df_hourly['IsHoliday'] = df_hourly['DayOfWeek'].isin([5, 6]).astype(int)
            df_hourly['IsPeak'] = (df_hourly['Global_active_power'] > df_hourly['Global_active_power'].quantile(0.75)).astype(int)
            
            df_hourly = self.remove_outliers(df_hourly, 'Global_active_power')
            
            self.data = df_hourly
            self.logger.info(f"Data loaded and resampled to hourly. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise

    def normalize_data(self, features, target):
        """
        Normalize the data using MinMaxScaler.
        Applies MinMax scaling to the specified feature columns and the target column together.
        Also fits a separate scaler for the target to allow inverse transformations later.

        Parameters
        ----------
        features : list of str
            List of column names to be used as features.
        target : str
            Column name of the target variable.
        """
        self.logger.info("Normalizing data")
        try:
            features = [f for f in features if f != target]
            features_scaled = self.feature_scaler.fit_transform(self.data[features])
            target_scaled = self.target_scaler.fit_transform(self.data[[target]])
            data_scaled = np.hstack((features_scaled, target_scaled))
            self.logger.info(f"Data normalized. Scaled range: min={data_scaled.min():.4f}, max={data_scaled.max():.4f}")
            return data_scaled, features, target
        except Exception as e:
            self.logger.error(f"Normalization failed: {str(e)}")
            raise

    def create_sequences(self, data, seq_length, forecast_horizon, n_features):
        """
        Generate input-output sequences for supervised time-series forecasting.

        This method creates sliding window sequences from the provided multivariate time-series data.
        Each input sequence (X) consists of `seq_length` time steps with all features, while the target 
        sequence (y) consists of the next `forecast_horizon` values of the first feature 
        (typically the main variable of interest, e.g., 'Global_active_power')

        Parameters
        ----------
        data : np.ndarray
            A 2D NumPy array of shape (n_samples, n_features), representing the time-series data.
        seq_length : int
            The number of past time steps to include in each input sequence.
        forecast_horizon : int
            The number of future steps to forecast (target output).
        """
        self.logger.info(f"Creating sequences with length {seq_length} and forecast horizon {forecast_horizon}")
        try:
            X, y = [], []
            for i in range(len(data) - seq_length - forecast_horizon + 1):
                X.append(data[i:i + seq_length, :n_features])
                y.append(data[i + seq_length:i + seq_length + forecast_horizon, -1])
            X, y = np.array(X), np.array(y)
            self.logger.info(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")
            return X, y
        except Exception as e:
            self.logger.error(f"Sequence creation failed: {str(e)}")
            raise
