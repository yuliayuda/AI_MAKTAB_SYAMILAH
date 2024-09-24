from transformers import pipeline

# Inisialisasi model penerjemahan (mT5 atau model lain yang mendukung Arabic-Indonesia)
translation_model = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en")

def translate_text(text, target_language="id"):
    # Terjemahkan dari Arab ke Indonesia atau sebaliknya
    if target_language == "id":
        result = translation_model(text, src_lang="ar", tgt_lang="id")
    else:
        result = translation_model(text, src_lang="id", tgt_lang="ar")
    
    return result[0]['translation_text']
