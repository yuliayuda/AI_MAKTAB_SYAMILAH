from modules.qa_module import get_answer
from modules.retrieval_module import search_reference
from utils.preprocessing import preprocess_input
from modules.translation_module import translate_text
from modules.summarization_module import summarize_text
from utils.postprocessing import format_output
from data.preprocessing_data import load_data, preprocess_data
from training.train_model import train_model
from evaluation.evaluate_model import evaluate_model
from logs.logging import setup_logging, log_message

def main():
    user_input = input("Masukkan pertanyaan: ")

    # Preprocessing input
    processed_input = preprocess_input(user_input)

    # Search for reference
    reference = search_reference(processed_input)
    
    # Get answer from QA model
    answer = get_answer(processed_input, reference)

    print("Jawaban:", answer)
    print("Referensi:", reference)


    # Terjemahkan:
    # Terjemahkan jawaban jika diperlukan
    translation = translate_text(answer, target_language="id")

    # Ringkas jawaban jika diperlukan
    summary = summarize_text(reference)

    # Format output untuk ditampilkan
    output = format_output(answer, reference, translation, summary)

print(output)

    # Preprocessing data():
    data = load_data('/kaggle/input/arabic-library/my_csv.csv')
    processed_data = preprocess_data(data)
    save_processed_data(processed_data, 'data/processed/processed_dataset.csv')
    
    # Misalkan Anda memiliki model yang sudah dilatih
    trained_model = train_model(model, processed_data)

    # Evaluasi model
    evaluation_results = evaluate_model(trained_model, processed_data)
print("Hasil evaluasi:", evaluation_results)

log_message(f"Hasil evaluasi: {evaluation_results}")


if __name__ == "__main__":
    main()
