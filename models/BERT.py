# models/BERT.py
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

import logging

# Mengatur konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Memulai proses pemrosesan data...")
    # Proses data dan pelatihan model

    logging.info("Mulai evaluasi model...")
    evaluation_results = evaluate_model(models, processed_data)
    
    # Cetak hasil evaluasi
    logging.info(f"Hasil evaluasi: {evaluation_results}")

if __name__ == "__main__":
    main()

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
    return model
