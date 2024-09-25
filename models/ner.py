from transformers import pipeline

def named_entity_recognition(text):
    ner_model = pipeline("ner", model="asafaya/bert-base-arabic")
    ner_results = ner_model(text)

    for entity in ner_results:
        print(f"Entity: {entity['word']}, Label: {entity['entity']}")
