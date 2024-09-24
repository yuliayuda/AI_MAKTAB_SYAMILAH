import pandas as pd

def load_data(file_path):
    # Memuat dataset dari file CSV
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Lakukan preprocessing pada data
    # Misalnya, menghapus nilai kosong atau duplikat
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df
