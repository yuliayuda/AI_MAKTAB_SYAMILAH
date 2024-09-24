# models/model_2.py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # 1 output for binary classification

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

def load_data(file_path):
    return pd.read_csv(file_path)

def train_model(data):
    texts = data['text'].values
    labels = data['label'].values

    # Preprocessing
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences)

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels_encoded, test_size=0.2)

    train_dataset = TextDataset(X_train, y_train)
    val_dataset = TextDataset(X_val, y_val)

    # Parameters
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    hidden_dim = 64

    # Model
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    for epoch in range(10):
        model.train()
        for texts, labels in DataLoader(train_dataset, batch_size=32):
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    data = load_data('/kaggle/input/arabic-library/my_csv.csv')
    train_model(data)
