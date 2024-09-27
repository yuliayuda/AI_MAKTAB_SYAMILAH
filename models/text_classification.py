from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from camel_tools.tokenizers.word import simple_word_tokenize

def train_text_classification(train_data, test_data):
    vectorizer = TfidfVectorizer(tokenizer=simple_word_tokenize)
    model = LogisticRegression()
    text  = "Sahih Muslim"

    pipeline = make_pipeline(vectorizer, model)
    pipeline.fit(train_data[text])

    accuracy = pipeline.score(test_data[text])
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return pipeline
