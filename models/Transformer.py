# models/Transformer.py
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

def load_data(file_path):
    return pd.read_csv(file_path)

def generate_text(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def train_model(dataset):
    # Tokenizer dan model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Tokenisasi dan pembuatan dataset
    encodings = tokenizer(list(dataset['text']), truncation=True, padding=True, max_length=512)
    labels = encodings['input_ids'].copy()  # Labels untuk pelatihan

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    dataset = TextDataset(encodings, labels)

    # Menentukan argumen pelatihan
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    return model

if __name__ == "__main__":
    data = load_data('/kaggle/input/arabic-library/my_csv.csv')
    prompt = "ما حكم اكل الميتة؟ "
    generated_text = generate_text(prompt)
    print(generated_text)
