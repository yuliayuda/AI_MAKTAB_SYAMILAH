import re
from camel_tools.utils.charmap import CharMapper
from nltk.tokenize import word_tokenize
import nltk

# Mengunduh tokenizer bahasa Arab
nltk.download('punkt')

def load_data(file_path):
    return pd.read_csv(file_path)

def arabic_preprocessing(text):
    # Gunakan mapper yang sesuai dari camel-tools (seperti bw2ar atau lainnya)
    mapper = CharMapper.builtin_mapper('bw2ar')  # Buckwalter to Arabic
    clean_text = mapper.map_string(text)
    
    # Tokenisasi menggunakan NLTK
    tokens = word_tokenize(clean_text)
    return " ".join(tokens)

def split_data(df, test_size=0.2):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=test_size)
    return train, test
