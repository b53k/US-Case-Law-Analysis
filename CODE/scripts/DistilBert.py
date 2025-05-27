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

# Verify required columns are present
required_columns = ['text', 'CombinedOutcomeText', 'winner']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {csv_file}.")


# Create "ModifiedText" by removing "CombinedOutcomeText" from "text"
def remove_combined_text(row):
    original = row['text']
    combined = row['CombinedOutcomeText']
    return original.replace(combined, "").strip()


df['ModifiedText'] = df.apply(remove_combined_text, axis=1)

# Map true winner labels to integers (e.g., "appellant": 0, "appellee": 1)
label_map = {"appellant": 0, "appellee": 1}
df['label'] = df['winner'].map(label_map)

print(f"Total cases: {len(df)}")

# -----------------------------
# 2. Sentence Embedding Setup Using ModifiedText
# -----------------------------
# We use a DistilBERT-based model for sentence embeddings.
embed_model_name = "sentence-transformers/distiluse-base-multilingual-cased"
sentence_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
sentence_model = AutoModel.from_pretrained(embed_model_name)
sentence_model.eval()  # Set model to evaluation mode


def get_sentence_embedding(sentence):
    """Compute the mean-pooled embedding for a given sentence."""
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
    Splits the document into sentences, computes embeddings for each,
    and pads/truncates to a fixed number (max_sentences).
    Returns a tensor of shape (max_sentences, hidden_dim).
    """
    sentences = sent_tokenize(document_text)
    embeddings = []
    for s in sentences[:max_sentences]:
        emb = get_sentence_embedding(s)
        embeddings.append(emb)
    # If no sentences found, default to a vector size of 512
    hidden_dim = embeddings[0].shape[0] if embeddings else 512
    # Pad with zeros if needed
    if len(embeddings) < max_sentences:
        pad = [torch.zeros(hidden_dim) for _ in range(max_sentences - len(embeddings))]
        embeddings.extend(pad)
    return torch.stack(embeddings)


print("Computing document embeddings from ModifiedText...")
document_embeddings = [get_document_embedding(text) for text in df['ModifiedText'].tolist()]
print("Document embeddings computed.")

# Stack all document embeddings: shape (N, max_sentences, input_dim)
X_all = torch.stack(document_embeddings)  # e.g., (887, 50, hidden_dim)
y_all = torch.tensor(df['label'].tolist())


# -----------------------------
# 3. Model Architecture: GRU-based Document Encoder + Classifier
# -----------------------------
class DocumentEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True):
        super(DocumentEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        # x: (batch, max_sentences, input_dim)
        _, h_n = self.gru(x)
        if self.gru.bidirectional:
            h = torch.cat((h_n[0], h_n[1]), dim=1)
        else:
            h = h_n[-1]
        return h


class WinningPartyClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super(WinningPartyClassifier, self).__init__()
        self.encoder = DocumentEncoder(input_dim, hidden_dim, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, x):
        doc_repr = self.encoder(x)
        logits = self.fc(doc_repr)
        return logits


input_dim = X_all.shape[2]  # e.g., 512 or 768
hidden_dim = 128
num_classes = 2
max_sentences = 50

model_classifier = WinningPartyClassifier(input_dim, hidden_dim, num_classes)
model_classifier.train()

# -----------------------------
# 4. Prepare Data for Training (Full Data) with Oversampling
# -----------------------------
# Split into train and test sets using full data (887 cases)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

# Create sample weights for oversampling on training set based on class frequency
class_counts = np.bincount(y_train.numpy())
class_weights = 1. / class_counts  # higher weight for minority class
sample_weights = [class_weights[label] for label in y_train.numpy()]
sample_weights = torch.DoubleTensor(sample_weights)
from torch.utils.data import WeightedRandomSampler

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)

# -----------------------------
# 5. Training the Model
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

print("\nClassification Report:")
report = classification_report(y_test.numpy(), predictions.numpy(), target_names=["appellant", "appellee"])
print(report)
