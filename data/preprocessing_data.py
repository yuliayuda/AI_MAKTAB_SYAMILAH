# data/preprocessing_data.py

import pandas as pd
import json

def load_data(file_path):
    # Memuat data dari file JSON
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame(data)  # Mengembalikan sebagai DataFrame

def preprocess_data(file_path):
    # Memuat data menggunakan load_data
    df = load_data(file_path)
    
    # Memproses DataFrame
    df.dropna(inplace=True)  # Menghapus baris dengan nilai NaN
    # Tambahkan langkah pemrosesan lain yang diperlukan

    return df
