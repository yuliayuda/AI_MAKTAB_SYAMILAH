# models/BERT.py
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

def load_data(file_path):
    return pd.read_csv(file_path)

def train_model(data):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenisasi
    encodings = tokenizer(list(data['text']), truncation=True, padding=True, max_length=512)
    labels = torch.tensor(data['label'].values)

    # Membuat dataset
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    dataset = TextDataset(encodings, labels)

    # Menyiapkan model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Menentukan argumen pelatihan
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Melatih model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

if __name__ == "__main__":
    data = load_data('/kaggle/input/arabic-library/my_csv.csv')
    train_model(data)
