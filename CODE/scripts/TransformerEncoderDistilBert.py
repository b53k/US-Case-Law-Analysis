import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# -----------------------------
# 1. Data Loading and Preprocessing
# -----------------------------
csv_file = 'caseLawRaw.csv'
df = pd.read_csv(csv_file)

# Verify required columns exist
required_columns = ['text', 'CombinedOutcomeText', 'winner']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {csv_file}.")


# Create "ModifiedText" by removing CombinedOutcomeText from text.
def remove_combined_text(row):
    original = row['text']
    combined = row['CombinedOutcomeText']
    return original.replace(combined, "").strip()


df['ModifiedText'] = df.apply(remove_combined_text, axis=1)

# Map true labels to integers: e.g., "appellant" -> 0, "appellee" -> 1.
label_map = {"appellant": 0, "appellee": 1}
df['label'] = df['winner'].map(label_map)

print(f"Total cases: {len(df)}")

# -----------------------------
# 2. Sentence Embedding Setup Using ModifiedText
# -----------------------------
# We use DistilBERT-based sentence transformer for sentence embeddings.
embed_model_name = "sentence-transformers/distiluse-base-multilingual-cased"
sentence_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
sentence_model = AutoModel.from_pretrained(embed_model_name)
sentence_model.eval()


def get_sentence_embedding(sentence):
    """Compute the mean-pooled embedding for a sentence."""
    inputs = sentence_tokenizer(sentence, truncation=True, padding=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = sentence_model(**inputs)
    token_embeddings = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
    sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    mean_embedding = sum_embeddings / sum_mask
    return mean_embedding.squeeze(0)  # (hidden_dim,)


def get_document_embedding(document_text, max_sentences=50):
    """
    Split the document (ModifiedText) into sentences, compute embeddings, and
    pad/truncate to a fixed number of sentences.
    Returns a tensor of shape (max_sentences, hidden_dim).
    """
    sentences = sent_tokenize(document_text)
    embeddings = []
    for s in sentences[:max_sentences]:
        emb = get_sentence_embedding(s)
        embeddings.append(emb)
    hidden_dim = embeddings[0].shape[0] if embeddings else 512
    if len(embeddings) < max_sentences:
        pad = [torch.zeros(hidden_dim) for _ in range(max_sentences - len(embeddings))]
        embeddings.extend(pad)
    return torch.stack(embeddings)


print("Computing document embeddings from ModifiedText...")
document_embeddings = [get_document_embedding(text) for text in df['ModifiedText'].tolist()]
print("Document embeddings computed.")

# Stack into a tensor: shape (N, max_sentences, input_dim)
X_all = torch.stack(document_embeddings)  # e.g., shape: (887, 50, hidden_dim)
y_all = torch.tensor(df['label'].tolist())


# -----------------------------
# 3. Transformer Encoder-based Model Architecture
# -----------------------------
# We define a document encoder using TransformerEncoder layers.
class DocumentTransformer(nn.Module):
    def __init__(self, input_dim, num_layers=1, nhead=8, dim_feedforward=512, dropout=0.1):
        super(DocumentTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim); Transformer expects (seq_len, batch, input_dim)
        x = x.transpose(0, 1)  # now shape: (seq_len, batch, input_dim)
        encoded = self.transformer_encoder(x)  # (seq_len, batch, input_dim)
        # Global average pooling over the sequence dimension:
        encoded = encoded.mean(dim=0)  # (batch, input_dim)
        return encoded


class WinningPartyClassifierTransformer(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(WinningPartyClassifierTransformer, self).__init__()
        self.encoder = DocumentTransformer(input_dim, num_layers=1, nhead=8, dim_feedforward=512, dropout=0.1)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: (batch, max_sentences, input_dim)
        doc_repr = self.encoder(x)
        logits = self.fc(doc_repr)
        return logits


input_dim = X_all.shape[2]  # e.g., 768 if using DistilBERT's embedding dimension
num_classes = 2
model_classifier = WinningPartyClassifierTransformer(input_dim, num_classes)
model_classifier.train()

# -----------------------------
# 4. Prepare Data for Training (Full Data) with Oversampling
# -----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

# Compute sample weights for oversampling on training set
class_counts = np.bincount(y_train.numpy())
class_weights = 1. / class_counts
sample_weights = [class_weights[label] for label in y_train.numpy()]
sample_weights = torch.DoubleTensor(sample_weights)
from torch.utils.data import WeightedRandomSampler

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)

# -----------------------------
# 5. Training the Transformer Encoder Model
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_classifier.parameters(), lr=1e-3)
num_epochs = 5

for epoch in range(num_epochs):
    model_classifier.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model_classifier(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# -----------------------------
# 6. Evaluation on Test Set
# -----------------------------
model_classifier.eval()
with torch.no_grad():
    test_logits = model_classifier(X_test)
    predictions = torch.argmax(test_logits, dim=1)

from sklearn.metrics import classification_report

print("\nClassification Report for Transformer Encoder (DistilBERT):")
report = classification_report(y_test.numpy(), predictions.numpy(), target_names=["appellant", "appellee"])
print(report)
