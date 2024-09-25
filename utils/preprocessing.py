import re
import pandas as pd
from tqdm import tqdm
from camel_tools.utils.charmap import CharMapper
from nltk.tokenize import word_tokenize
import nltk

# Mengunduh tokenizer bahasa Arab
nltk.download('punkt')

def arabic_preprocessing(texts):
    # Menggunakan tqdm untuk menampilkan progres pemrosesan teks
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
    
    # Menghitung total baris di CSV
    total_rows = sum(1 for _ in open(file_path)) - 1  
    chunk_results = []

    # Menggunakan tqdm untuk memproses setiap chunk
    for chunk in tqdm(pd.read_csv(file_path, chunksize=chunksize), total=(total_rows // chunksize), desc="Memproses chunk", unit="chunk"):
        # Proses teks dalam chunk
        processed_texts = arabic_preprocessing(chunk['text'])
        chunk_results.extend(processed_texts)

    print("\nSemua chunk diproses.")
    return chunk_results



# Contoh penggunaan
file_path = "/kaggle/input/arabic-library/my_csv.csv"  # Ganti dengan path dataset Anda
processed_data = load_data_in_chunks(file_path, chunksize=1000)

# Menyimpan hasil ke file CSV baru jika diperlukan
pd.DataFrame(processed_data, columns=['processed_text']).to_csv("data/processed/processed_data.csv", index=False)
