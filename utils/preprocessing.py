import re
import pandas as pd
from tqdm import tqdm
from camel_tools.utils.charmap import CharMapper
from nltk.tokenize import word_tokenize
import nltk

# Mengunduh tokenizer bahasa Arab
nltk.download('punkt')

def arabic_preprocessing(texts):
    tqdm.pandas(desc="Memproses teks")

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

def load_data_in_chunks(file_path, chunksize=1000):
    print("Membaca dataset dalam chunks...")
    
    total_rows = sum(1 for _ in open(file_path)) - 1  # Menghitung total baris di CSV
    chunk_results = []
    total_chunks = 0

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        total_chunks += 1
        chunk_size = len(chunk)
        
        # Tampilkan informasi chunk yang sedang diproses
        print(f"\nMemproses chunk: ke {total_chunks} dengan {chunk_size} baris...", end='\r')
        
        # Proses teks dalam chunk
        processed_texts = arabic_preprocessing(chunk['text'])
        chunk_results.extend(processed_texts)

        # Update progress
        processed_rows = total_chunks * chunksize
        percentage_loaded = (processed_rows / total_rows) * 100
        progress_bar = '>' * int(percentage_loaded // 5) + '-' * (20 - int(percentage_loaded // 5))
        print(f"Load Datasets: [{progress_bar}] {percentage_loaded:.2f}% ({processed_rows}/{total_rows})", end='\r')

    print("\n\nSemua chunk diproses.")
    return chunk_results




# Contoh penggunaan
file_path = "/kaggle/input/arabic-library/my_csv.csv"  # Ganti dengan path dataset Anda
processed_data = load_data_in_chunks(file_path, chunksize=1000)

# Menyimpan hasil ke file CSV baru jika diperlukan
pd.DataFrame(processed_data, columns=['processed_text']).to_csv("data/processed/processed_data.csv", index=False)
