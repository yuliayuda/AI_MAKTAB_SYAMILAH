# main.py

# Impor pustaka yang diperlukan
import pandas as pd
import logging  # Untuk logging

# Impor dari modul Anda
from data.preprocessing_data import load_data, preprocess_data
from training.train_model import train_all_models
from evaluation.evaluate_model import evaluate_model
from logs.logging import setup_logging

def main():
    # Setup logging
    setup_logging()
    
    # Proses memuat dan mempersiapkan data
    logging.info("Memulai proses pemrosesan data...")
    data = load_data('/kaggle/input/ar-id-en-stop/ar_stop_word.json')  # Ganti dengan path data Anda
    processed_data = preprocess_data(data)
    processed_file_path = 'data/processed/processed_dataset.csv'  # Pastikan direktori ini ada
    processed_data.to_csv(processed_file_path, index=False)  # Simpan sebagai CSV
    logging.info("Pemrosesan data selesai.")

    # Memulai pelatihan model
    logging.info("Mulai proses pelatihan...")
    try:
        models = train_all_models(processed_data)  # Memanggil fungsi pelatihan
        logging.info("Pelatihan selesai untuk semua model.")
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat pelatihan: {e}")

    # Evaluasi model
    evaluation_results = evaluate_model(models, processed_data)  # Sesuaikan dengan parameter yang tepat
    logging.info(f"Hasil evaluasi: {evaluation_results}")

if __name__ == "__main__":
    main()
