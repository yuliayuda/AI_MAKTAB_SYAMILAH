import re
import pandas as pd
from tqdm import tqdm  # Library untuk progress bar
from camel_tools.utils.charmap import CharMapper
from nltk.tokenize import word_tokenize
import nltk

# Mengunduh tokenizer bahasa Arab
nltk.download('punkt')

# Menambahkan progress bar saat membaca data
def load_data(file_path):
    print("Membaca dataset...")
    return pd.read_csv(file_path)

# Menambahkan progress bar saat preprocessing teks Arab
def arabic_preprocessing(texts):
    tqdm.pandas(desc="Memproses teks")
    
    # Fungsi untuk membersihkan dan memproses teks
    def clean_text(text):
        # Gunakan mapper yang sesuai dari camel-tools (seperti bw2ar atau lainnya)
        mapper = CharMapper.builtin_mapper('bw2ar')  # Buckwalter to Arabic
        clean_text = mapper.map_string(text)
        
        # Tokenisasi menggunakan NLTK
        tokens = word_tokenize(clean_text)
        return " ".join(tokens)

    # Menggunakan progress_apply untuk menunjukkan progress bar
    return texts.progress_apply(clean_text)

# Menambahkan progress bar saat membagi data
def split_data(df, test_size=0.2):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=test_size)
    return train, test
