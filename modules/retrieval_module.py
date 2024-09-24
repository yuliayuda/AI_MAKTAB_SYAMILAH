# Misalnya kita menggunakan ElasticSearch atau model BM25
from elasticsearch import Elasticsearch

BONSAI_URL = "https://68f8ab27wj:i1os1l6puq@spruce-25009067.us-east-1.bonsaisearch.net:443"
es = Elasticsearch(BONSAI_URL)

def search_reference(query):
    # Contoh query sederhana
    res = es.search(index="islamic_literature", body={"query": {"match": {"text": query}}})
    hits = res['hits']['hits']
    
    if hits:
        return hits[0]['_source']['text']
    return "Referensi tidak ditemukan."
