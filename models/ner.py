from transformers import pipeline

def named_entity_recognition(text):
      text  = "Sahih Muslim"
      ner_model = pipeline("ner", model="asafaya/bert-base-arabic")
      retrieved_text = ner_model(text)

for entity in retrieved_text:
      print(f"Entity: {entity['word']}, Label: {entity['entity']}")



