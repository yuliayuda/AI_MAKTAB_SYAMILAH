# main.py

import logging
from data.preprocessing_data import preprocess_data
from training.train_model import train_all_models
from evaluation.evaluate_model import evaluate_model

def main():
    # Mengatur konfigurasi logging
    logging.basicConfig(level=logging.INFO)

    # Proses memuat dan mempersiapkan data
    logging.info("Memulai proses pemrosesan data...")
    data = preprocess_data('/kaggle/input/ar-id-en-stop/ar_stop_word.json')  # Path data JSON
    logging.info("Pemrosesan data selesai.")

    # Inisialisasi models
    models = None

    # Memulai pelatihan model
    logging.info("Mulai proses pelatihan...")
    try:
        models = train_all_models(data)  # Memanggil fungsi pelatihan
        logging.info("Pelatihan selesai untuk semua model.")
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat pelatihan: {e}")

    # Hanya evaluasi jika models tidak None
    if models is not None:
        logging.info("Memulai evaluasi model...")
        evaluation_results = evaluate_model(models, data)  # Sesuaikan dengan parameter yang tepat
        logging.info("Hasil evaluasi: {}".format(evaluation_results))
    else:
        logging.warning("Model tidak dilatih, tidak ada hasil evaluasi yang dapat dilakukan.")

if __name__ == "__main__":
    main()
