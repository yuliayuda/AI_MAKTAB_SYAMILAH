import re
import pandas as pd
from tqdm import tqdm
from camel_tools.utils.charmap import CharMapper
from nltk.tokenize import word_tokenize
import nltk
from sklearn.model_selection import train_test_split  # Tambahkan ini

# Mengunduh tokenizer bahasa Arab
nltk.download('punkt')

def arabic_preprocessing(texts):
    tqdm.pandas(desc="Memproses teks")
    
    def clean_text(text):
        mapper = CharMapper.builtin_mapper('bw2ar')  # Buckwalter to Arabic
        clean_text = mapper.map_string(text)
        
        tokens = word_tokenize(clean_text)
        return " ".join(tokens)

    # Menggunakan tqdm untuk memproses teks
    processed_texts = []
    for text in tqdm(texts, desc="Memproses teks", unit="teks"):
        processed_texts.append(clean_text(text))
    
    return processed_texts

def load_data_in_chunks(file_path, chunksize=1000):
    print("Membaca dataset...")
    
    chunk_results = []
    
    for chunk in tqdm(pd.read_csv(file_path, chunksize=chunksize), desc="Memproses chunk", unit="chunk"):
        processed_texts = arabic_preprocessing(chunk['text'])
        chunk_results.extend(processed_texts)

    print("\nSemua chunk diproses.")
    return pd.DataFrame(chunk_results, columns=['text'])  # Mengembalikan DataFrame

def split_data(df, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_data, test_data
