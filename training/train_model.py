# training/train_model.py
from models.BERT import train_model as train_bert
from models.LSTM import train_model as train_lstm
from models.Transformer import train_model as train_transformer

def train_all_models(data):
    bert_model = train_bert(data)
    lstm_model = train_lstm(data)
    transformer_model = train_transformer(data)
    
    return bert_model, lstm_model, transformer_model
