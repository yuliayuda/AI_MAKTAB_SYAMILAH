# models/model_3.py
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_data(file_path):
    return pd.read_csv(file_path)

def generate_text(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    data = load_data('/kaggle/input/arabic-library/my_csv.csv')
    prompt = "ما حكم اكل الميتة؟ "
    generated_text = generate_text(prompt)
    print(generated_text)
