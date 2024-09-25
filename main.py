# main.py

# Impor pustaka yang diperlukan
import pandas as pd
import numpy as np
from transformers import pipeline  # Jika Anda menggunakan fungsi dari transformers
import torch  # Jika Anda menggunakan PyTorch
import tensorflow as tf  # Jika Anda menggunakan TensorFlow
from elasticsearch import Elasticsearch  # Jika Anda menggunakan Elasticsearch

# Impor dari modul Anda
from modules.qa_module import get_answer
from modules.retrieval_module import search_reference
from utils.preprocessing import preprocess_input
from modules.translation_module import translate_text
from modules.summarization_module import summarize_text
from utils.postprocessing import format_output
from data.preprocessing_data import load_data, preprocess_data
from training.train_model import train_all_models
from evaluation.evaluate_model import evaluate_model
from logs.logging import setup_logging, log_message


def main():
    # Setup logging (jika diperlukan)
    setup_logging()

    user_input = input("حكم النية في الصلاة؟")

    # Preprocessing input
    processed_input = preprocess_input(user_input)

    # Search for reference
    reference = search_reference(processed_input)
    
    # Get answer from QA model
    answer = get_answer(processed_input, reference)

    print("الجواب:", answer)  # "Jawaban:" dalam bahasa Arab
    print("المصدر:", reference)  # "Referensi:" dalam bahasa Arab

    # Terjemahkan jawaban jika diperlukan
    translation = translate_text(answer, target_language="id")

    # Ringkas jawaban jika diperlukan
    summary = summarize_text(reference)

    # Format output untuk ditampilkan
    output = format_output(answer, reference, translation, summary)

    print("Output:", output)

    # Preprocessing data
    data = load_data('/kaggle/input/arabic-library/my_csv.csv')
    processed_data = preprocess_data(data)
    processed_file_path = 'data/processed/processed_dataset.csv'  # Pastikan direktori ini ada
    processed_data.to_csv(processed_file_path, index=False)  # Simpan sebagai CSV

    # Misalkan Anda memiliki model yang sudah dilatih
    trained_model = train_model(processed_data)

    # Evaluasi model
    evaluation_results = evaluate_model(trained_model, processed_data)
    print("Hasil evaluasi:", evaluation_results)

    log_message(f"Hasil evaluasi: {evaluation_results}")


if __name__ == "__main__":
    main()
train_all_models(data)
