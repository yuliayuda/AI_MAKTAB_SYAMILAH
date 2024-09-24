# Konfigurasi dasar untuk model-model dan pipeline

config = {
    "qa_model": "CAMeL-Lab/bert-base-arabic-camelbert-da",
    "retrieval_index": "islamic_literature",
    "translation_model": "Helsinki-NLP/opus-mt-ar-en",
    "summarization_model": "t5-small"
}

def get_config():
    return config
