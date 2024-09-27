from transformers import pipeline

def retrieve_information(text):
      text  = "Sahih Muslim"
      ner_model = pipeline("ner", model="asafaya/bert-base-arabic")
      ner_results = ner_model(text)

for entity in ner_results:
      print(f"Entity: {entity['word']}, Label: {entity['entity']}")



