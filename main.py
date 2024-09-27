import pandas as pd
from utils.preprocessing import load_data, arabic_preprocessing
from models.text_classification import train_text_classification
from models.information_retrieval import retrieve_information
from models.ner import named_entity_recognition
from models.question_answering import answer_question 
from sklearn.model_selection import train_test_split
import joblib  # Untuk menyimpan model

# Load and preprocess Arabic dataset
train_data = load_data("/kaggle/input/arabic-hadith/All Hadith Books/Sahih Bukhari.csv")
train_data['Sahih Bukhari'] = arabic_preprocessing(train_data['Sahih Bukhari'])

# Load and preprocess Arabic dataset for testing from different CSV file
test_data = load_data("/kaggle/input/arabic-hadith/All Hadith Books/Sahih Muslim.csv")
text = "Sahih Muslim"
test_data[text] = arabic_preprocessing(test_data[text])

# 1. Text Classification
print("Training Text Classification model...")
text_class_model = train_text_classification(train_data, test_data)

# Simpan model ke file
joblib.dump(text_class_model, 'text_classification_model.pkl')

# 2. Information Retrieval
print("\nInformation Retrieval Test:")
query = "ما هو الموضوع الرئيسي في هذا النص؟"
retrieved_text = retrieve_information(query, train_data)
print("Retrieved Text:", retrieved_text)

# Simpan teks yang diambil ke dalam CSV
retrieved_df = pd.DataFrame({"Retrieved_Text": [retrieved_text]})
retrieved_df.to_csv('retrieved_text.csv', index=False)

# 3. Named Entity Recognition (NER)
print("\nNamed Entity Recognition Test:")
ner_results = named_entity_recognition(retrieved_text)
print("NER Results:", ner_results)

# Simpan hasil NER ke dalam file
ner_results_df = pd.DataFrame(ner_results)
ner_results_df.to_csv('ner_results.csv', index=False)

# 4. Question Answering
print("\nQuestion Answering Test:")
context = retrieved_text
question = "ما هو هذا النص حول؟"
answer = answer_question(question, context)
print("Answer:", answer)

# Simpan jawaban ke dalam file
with open('answer.txt', 'w', encoding='utf-8') as f:
    f.write(answer)
