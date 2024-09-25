from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from camel_tools.tokenizers.word import simple_word_tokenize

def train_text_classification(train_data, test_data):
    # Tokenizer Arab
    vectorizer = TfidfVectorizer(tokenizer=simple_word_tokenize)
    model = LogisticRegression()

    pipeline = make_pipeline(vectorizer, model)
    pipeline.fit(train_data['text'], train_data['Book_name'])

    accuracy = pipeline.score(test_data['text'], test_data['Book_name'])
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return pipeline
