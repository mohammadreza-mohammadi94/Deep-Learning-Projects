# from src.preprocess import DataPreprocessor
# from src.models import rnn_model, lstm_model
# from src.train_evaluate import train_model, evaluate_and_plot
# from src.utils import setup_logging
# import numpy as np
# import tensorflow as tf

# def main():
#     # Setup logger
#     logger = setup_logging()

#     # Setup parameters
#     MAX_WORDS = 10000
#     MAX_LEN = 200
#     N_EPOCHS = 30
#     BATCH_SIZE = 128

#     # Loading and preprocessing data
#     preprocessor = DataPreprocessor(logger,
#                                     max_words=MAX_WORDS,
#                                     max_len=MAX_LEN)
#     X_train, X_test, y_train, y_test = preprocessor.load_data()

#     logger.info("Class distribution in training data:")
#     logger.info(f"Negative (0): {np.sum(y_train == 0)}, Positive (1): {np.sum(y_train == 1)}")
#     logger.info("Class distribution in test data:")
#     logger.info(f"Negative (0): {np.sum(y_test == 0)}, Positive (1): {np.sum(y_test == 1)}")

#     # Define models
#     models = [
#         ('RNN', rnn_model(vocab_size=MAX_WORDS, embedding_dim=100, max_len=MAX_LEN)),
#         ('LSTM', lstm_model(vocab_size=MAX_WORDS, embedding_dim=100, max_len=MAX_LEN))
#     ]

#     # Fitting models
#     histories = []
#     for name, model in models:
#         history = train_model(model, X_train, y_train,
#                               X_test, y_test,
#                               N_EPOCHS, BATCH_SIZE, logger)
#         histories.append(history)

#     # Evaluation and plotting
#     evaluate_and_plot([model for _, model in models], histories, X_test, y_test, 
#                       preprocessor.get_class_names(), logger)
    
#     logger.info("Execution completed successfully")

# if __name__ == "__main__":
#     main()

import tensorflow as tf
tf.config.run_functions_eagerly(True)  # فعال‌سازی Eager Execution
from src.preprocess import DataPreprocessor
from src.models import BertSentimentModel
from src.utils import setup_logging
import numpy as np
def main():
    logger = setup_logging()
    MAX_WORDS = 10000
    MAX_LEN = 200
    N_EPOCHS = 3
    BATCH_SIZE = 8

    # بارگذاری و پیش‌پردازش داده‌ها
    preprocessor = DataPreprocessor(logger, max_words=MAX_WORDS, max_len=MAX_LEN)
    X_train, X_test, y_train, y_test = preprocessor.load_data()

    # بررسی تعادل کلاس‌ها
    logger.info("Class distribution in training data:")
    logger.info(f"Negative (0): {np.sum(y_train == 0)}, Positive (1): {np.sum(y_train == 1)}")
    logger.info("Class distribution in test data:")
    logger.info(f"Negative (0): {np.sum(y_test == 0)}, Positive (1): {np.sum(y_test == 1)}")

    # نمایش چند نمونه از داده‌ها
    word_index = tf.keras.datasets.imdb.get_word_index()
    bert_model = BertSentimentModel(logger, max_len=MAX_LEN)
    logger.info("Sample training reviews:")
    for i in range(3):
        logger.info(f"Review {i}: {bert_model.decode_review(X_train[i], word_index)}")
        logger.info(f"Label {i}: {y_train[i]}")

    # تعریف و آموزش مدل BERT
    train_dataset, test_dataset, train_encodings, test_encodings = bert_model.preprocess_data(X_train, X_test, y_train, y_test)
    history = bert_model.train(train_dataset, test_dataset, N_EPOCHS, BATCH_SIZE)
    bert_model.evaluate_and_plot(history, test_encodings, y_test)

    logger.info("Execution completed successfully")

if __name__ == "__main__":
    main()