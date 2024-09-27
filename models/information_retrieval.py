from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from camel_tools.tokenizers.word import simple_word_tokenize

def retrieve_information(query, data):
    text  = "Sahih Muslim"
    vectorizer = TfidfVectorizer(tokenizer=simple_word_tokenize)
    tfidf_matrix = vectorizer.fit_transform(data[text])
    
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    top_index = similarities.argmax()
    return data.iloc[top_index][text]
