import torch
from torch import nn
from transformers import BertModel, BertForTokenClassification

class Bert_BiDirectional_LSTM(nn.Module):
    def __init__(self, bert_model_name, num_labels, hidden_dim=768, lstm_layers=1, dropout=0.1):
        super(Bert_BiDirectional_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=lstm_layers, 
                            bidirectional=True, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        logits = self.fc(lstm_output)
        return logits

class BertEntityRecognizer(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertEntityRecognizer, self).__init__()
        self.model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
