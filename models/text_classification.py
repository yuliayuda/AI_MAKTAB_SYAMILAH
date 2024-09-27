from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from camel_tools.tokenizers.word import simple_word_tokenize

def train_text_classification(train_data, test_data):
    vectorizer = TfidfVectorizer(tokenizer=simple_word_tokenize)
    model = LogisticRegression()
    text = "Sahih Bukhari"
    book_name = "Sahih Muslim"

    pipeline = make_pipeline(vectorizer, model)
    pipeline.fit(train_data[text],train_data[book_name])

    accuracy = pipeline.score(test_data[book_name],test_data[text])
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return pipeline
