import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

class BertSentimentModel:
    def __init__(self, logger, max_len=200, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.logger = logger
        self.max_len = max_len
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(self.device)
        self.class_names = ['Negative', 'Positive']

    def decode_review(self, text, word_index):
        """
        Decode a review from indices to text.
        """
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

    def preprocess_data(self, X_train, X_test, y_train, y_test):
        """
        Preprocess the IMDB dataset for BERT.
        """
        self.logger.info("Preprocessing data for BERT")
        word_index = tf.keras.datasets.imdb.get_word_index()
        X_train_texts = [self.decode_review(x, word_index) for x in X_train]
        X_test_texts = [self.decode_review(x, word_index) for x in X_test]

        train_encodings = self.tokenizer(X_train_texts, truncation=True, padding=True, max_length=self.max_len, return_tensors='pt')
        test_encodings = self.tokenizer(X_test_texts, truncation=True, padding=True, max_length=self.max_len, return_tensors='pt')

        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(y_train, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            test_encodings['input_ids'],
            test_encodings['attention_mask'],
            torch.tensor(y_test, dtype=torch.long)
        )

        return train_dataset, test_dataset, train_encodings, test_encodings

    def train(self, train_dataset, test_dataset, epochs, batch_size):
        """
        Train the BERT model.
        """
        self.logger.info("Training BERT model")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch in train_loader:
                input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = total_correct / total_samples
            history['loss'].append(avg_train_loss)
            history['accuracy'].append(train_accuracy)

            # ارزیابی روی داده‌های تست
            self.model.eval()
            total_val_loss = 0
            total_val_correct = 0
            total_val_samples = 0

            with torch.no_grad():
                for batch in test_loader:
                    input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                    total_val_loss += loss.item()
                    preds = torch.argmax(logits, dim=1)
                    total_val_correct += (preds == labels).sum().item()
                    total_val_samples += labels.size(0)

            avg_val_loss = total_val_loss / len(test_loader)
            val_accuracy = total_val_correct / total_val_samples
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)

            self.logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        return history

    def evaluate_and_plot(self, history, test_encodings, y_test):
        """
        Evaluate the BERT model and generate plots.
        """
        self.logger.info("Evaluating BERT model and generating plots")
        self.model.eval()
        input_ids = test_encodings['input_ids'].to(self.device)
        attention_mask = test_encodings['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()

        accuracy = accuracy_score(y_test, y_pred)
        self.logger.info(f"BERT Accuracy: {accuracy:.4f}")
        print(f"BERT Accuracy: {accuracy:.4f}\n")
        print("BERT Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix: BERT')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('sentiment_confusion_matrix_bert.png')
        plt.close()
        self.logger.info("Confusion matrix for BERT saved as 'sentiment_confusion_matrix_bert.png'")

        plt.figure(figsize=(10, 5))
        plt.plot(history['loss'], label='Training Loss', color='blue')
        plt.plot(history['val_loss'], label='Validation Loss', color='orange', linestyle='--')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('sentiment_loss_history_bert.png')
        plt.close()
        self.logger.info("Loss history plot saved as 'sentiment_loss_history_bert.png'")

        plt.figure(figsize=(10, 5))
        plt.plot(history['accuracy'], label='Training Accuracy', color='blue')
        plt.plot(history['val_accuracy'], label='Validation Accuracy', color='orange', linestyle='--')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('sentiment_accuracy_history_bert.png')
        plt.close()
        self.logger.info("Accuracy history plot saved as 'sentiment_accuracy_history_bert.png'")