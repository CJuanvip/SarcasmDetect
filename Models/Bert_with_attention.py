import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

class SarcasmDataset(Dataset):
    """Custom PyTorch Dataset class to handle tokenized data for input into a BERT model."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class AttentionHead(nn.Module):
    """Attention mechanism to weigh the importance of different tokens across the sequence."""
    def __init__(self, dimensions, attention_head_size):
        super(AttentionHead, self).__init__()
        self.dimensions = dimensions
        self.attention_head_size = attention_head_size
        self.W = nn.Linear(dimensions, attention_head_size)
        self.V = nn.Linear(attention_head_size, 1, bias=False)

    def forward(self, features):
        # Attending to the features
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        attended_features = attention_weights * features
        return attended_features.sum(dim=1)

class BERTWithAttention(nn.Module):
    """Custom BERT model with an additional attention head."""
    def __init__(self, bert_model_path):
        super(BERTWithAttention, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.attention = AttentionHead(768, 100)  # Adjust sizes as necessary
        self.classifier = nn.Linear(768, 2)  # Binary classification
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        attended = self.attention(last_hidden_state)
        dropped = self.dropout(attended)
        logits = self.classifier(dropped)
        return logits

def compute_metrics(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    accuracy = accuracy_score(labels_flat, preds_flat)
    f1 = f1_score(labels_flat, preds_flat)
    return {"accuracy": accuracy, "f1_score": f1}

# Example usage
if __name__ == '__main__':
    df = pd.read_csv('path_to_train_dataset.csv')
    df['labels'] = df['sarcastic'].apply(lambda x: 1 if x else 0)

    sentences = df['tweet'].values
    labels = df['labels'].values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTWithAttention('bert-base-uncased')

    # Tokenization
    encodings = tokenizer(list(sentences), truncation=True, padding=True, max_length=128)

    # Dataset and DataLoader
    dataset = SarcasmDataset(encodings, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Example training loop setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training loop would go here

    # Evaluate the model
    # Evaluation code would follow
