import pandas as pd
from camel_tools.utils.charmap import CharMapper
from nltk.tokenize import word_tokenize
import nltk

# Mengunduh tokenizer bahasa Arab
nltk.download('punkt')

def load_data(file_path):
    return pd.read_csv(file_path)

def arabic_preprocessing(text):
    # Normalisasi teks Arab menggunakan CharMapper dari camel-tools
    mapper = CharMapper.builtin_mapper('ar_clean')
    clean_text = mapper.map_string(text)
    
    # Tokenisasi
    tokens = word_tokenize(clean_text)
    return " ".join(tokens)

def split_data(df, test_size=0.2):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=test_size)
    return train, test
