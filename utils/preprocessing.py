import re
import pandas as pd
from tqdm import tqdm
from camel_tools.utils.charmap import CharMapper
from nltk.tokenize import word_tokenize
import nltk

# Mengunduh tokenizer bahasa Arab
nltk.download('punkt')

def load_data(file_path):
    print("Membaca dataset...")
    return pd.read_csv(file_path)

def arabic_preprocessing(texts):
    tqdm.pandas(desc="Memproses teks")

    # Fungsi untuk membersihkan dan memproses teks
    def clean_text(text):
        mapper = CharMapper.builtin_mapper('bw2ar')  # Buckwalter to Arabic
        clean_text = mapper.map_string(text)
        
        tokens = word_tokenize(clean_text)
        return " ".join(tokens)

    processed_texts = []
    total = len(texts)

    for index, text in enumerate(texts):
        processed_texts.append(clean_text(text))
        # Hitung dan tampilkan persentase
        percentage = (index + 1) / total * 100
        print(f"Memproses teks: {percentage:.2f}% selesai", end='\r')  # Menampilkan progres di satu baris

    print("\nProses selesai.")
    return processed_texts

def split_data(df, test_size=0.2):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=test_size)
    return train, test
