# Misalnya kita menggunakan ElasticSearch atau model BM25
from elasticsearch import Elasticsearch

es = Elasticsearch()

def search_reference(query):
    # Contoh query sederhana
    res = es.search(index="islamic_literature", body={"query": {"match": {"text": query}}})
    hits = res['hits']['hits']
    
    if hits:
        return hits[0]['_source']['text']
    return "Referensi tidak ditemukan."