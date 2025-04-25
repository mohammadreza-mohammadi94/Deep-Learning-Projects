# import tensorflow as tf
# import numpy as np


# class DataPreprocessor:
#     def __init__(self, logger, max_words=10000, max_len=100):
#         self.logger = logger
#         self.max_words = max_words
#         self.max_len = max_len
#         self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)

#     def load_data(self):
#         """
#         Load and preprocess the IMDB dataset.

#         Returns
#         -------
#         X_train, X_test, y_train, y_test : tuple of np.ndarray
#             Preprocessed training and testing data.
#         """

#         self.logger.info("Loading and preprocessing IMDB dataset")
#         # loading data
#         (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
#                                                         num_words=self.max_words,
#                                                         )

#         # Padding sequences
#         # Padding is usefull for equalizing sequences (to have the same length)
#         X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.max_len)
#         X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=self.max_len)

#         self.logger.info(f"Data loaded and preprorcessed. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#         return X_train, X_test, y_train, y_test
    
#     def get_class_names(self):
#         """
#         Return the names of classes
#         """
#         return ['Negative', 'Positive']

import tensorflow as tf
import numpy as np

class DataPreprocessor:
    def __init__(self, logger, max_words=10000, max_len=100):
        self.logger = logger
        self.max_words = max_words  # حداکثر تعداد کلمات در واژگان
        self.max_len = max_len      # حداکثر طول دنباله
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)

    def load_data(self):
        """
        Load and preprocess the IMDB dataset.

        Returns
        -------
        X_train, X_test, y_train, y_test : tuple of np.ndarray
            Preprocessed training and testing data.
        """
        self.logger.info("Loading and preprocessing IMDB dataset")
        # بارگذاری داده‌ها
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
            num_words=self.max_words,
            start_char=None,
            oov_char=2,
            index_from=3
        )

        # تنظیم اندیس‌ها برای شروع از 0
        X_train = [[max(w - 3, 0) if w >= 3 else w for w in x] for x in X_train]
        X_test = [[max(w - 3, 0) if w >= 3 else w for w in x] for x in X_test]

        # محدود کردن اندیس‌ها به بازه [0, max_words)
        X_train = [[w if w < self.max_words else 2 for w in x] for x in X_train]
        X_test = [[w if w < self.max_words else 2 for w in x] for x in X_test]

        # پد کردن دنباله‌ها برای داشتن طول یکسان
        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.max_len)
        X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=self.max_len)

        self.logger.info(f"Data loaded and preprocessed. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def get_class_names(self):
        """
        Return the names of the classes.
        """
        return ['Negative', 'Positive']
