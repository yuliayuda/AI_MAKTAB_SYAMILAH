from datasets import load_metric
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

def evaluate_model(model, dataset):
    metric = load_metric("accuracy")

    predictions = model.predict(dataset['validation'])
    preds = predictions.predictions.argmax(-1)
    metric.add_batch(predictions=preds, references=dataset['validation']['labels'])

    result = metric.compute()
    return result
