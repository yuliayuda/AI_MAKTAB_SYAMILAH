# training/train_model.py
from models.BERT import train_model as train_bert
from models.LSTM import train_model as train_lstm
from models.Transformer import train_model as train_transformer
import logging

# Mengatur konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Memulai proses pemrosesan data...")
    # Proses data dan pelatihan model

    logging.info("Mulai evaluasi model...")
    evaluation_results = evaluate_model(models, processed_data)
    
    # Cetak hasil evaluasi
    logging.info(f"Hasil evaluasi: {evaluation_results}")

if __name__ == "__main__":
    main()


def train_all_models(data):
    bert_model = train_bert(data)
    lstm_model = train_lstm(data)
    transformer_model = train_transformer(data)
    
    return bert_model, lstm_model, transformer_model
