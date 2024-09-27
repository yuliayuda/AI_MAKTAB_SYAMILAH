from utils.preprocessing import load_data, arabic_preprocessing
from models.text_classification import train_text_classification
from models.information_retrieval import retrieve_information
from models.ner import named_entity_recognition
from models.question_answering import answer_question

# Load and preprocess Arabic dataset
df = load_data("/kaggle/input/arabic-hadith/All Hadith Books/Sahih Bukhari.csv")
df['text'] = arabic_preprocessing(df['text'])

# 1. Text Classification
print("Training Text Classification model...")
text_class_model = train_text_classification(df)  # Sesuaikan jika diperlukan

# 2. Information Retrieval
print("\nInformation Retrieval Test:")
query = "ما هو الموضوع الرئيسي في هذا النص؟"
retrieved_text = retrieve_information(query, df)
print("Retrieved Text:", retrieved_text)

# 3. Named Entity Recognition (NER)
print("\nNamed Entity Recognition Test:")
named_entity_recognition(retrieved_text)

# 4. Question Answering
print("\nQuestion Answering Test:")
context = retrieved_text
question = "ما هو هذا النص حول؟"
answer = answer_question(question, context)
print("Answer:", answer)
