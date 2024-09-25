from transformers import pipeline

def answer_question(question, context):
    # Menggunakan model BERT khusus untuk bahasa Arab
    qa_model = pipeline("question-answering", model="asafaya/bert-base-arabic")
    result = qa_model(question=question, context=context)
    return result['answer']
