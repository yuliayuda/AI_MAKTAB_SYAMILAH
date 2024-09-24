from transformers import pipeline

# Inisialisasi model question answering (BERT, AraBERT, atau lainnya)
qa_model = pipeline("question-answering", model="CAMeL-Lab/bert-base-arabic-camelbert-da", device=0)  # device=0 for GPU)


def get_answer(question, context):
    response = qa_model(question=question, context=context)
    return response['answer']
