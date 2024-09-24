from transformers import pipeline

# Inisialisasi model summarization (T5 atau mT5)
summarization_model = pipeline("summarization", model="t5-small", device=0)  # device=0 for GPU))

def summarize_text(text):
    # Buat ringkasan teks
    summary = summarization_model(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']
