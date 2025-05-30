"""
Dataset: https://en.wikipedia.org/wiki/Max_Payne
"""

import re
import numpy as np
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

MAX_VOCAB_SIZE = 5000
MAX_SEQ_LEN = 200
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

url = "https://en.wikipedia.org/wiki/Max_Payne"
print("Δεδομένα από URL:", url)
response = requests.get(url)
if response.status_code != 200:
    raise Exception("Αποτυχία")
html_content = response.text

soup = BeautifulSoup(html_content, "html.parser")
paragraphs = soup.find_all("p")
print(f"Βρέθηκαν {len(paragraphs)} παραγράφοι")

texts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50]
print(f"Χρησιμοποιούνται {len(texts)} παραγράφοι ως δείγματα.")

labels = [0 if i % 2 == 0 else 1 for i in range(len(texts))]
print("Παραδείγματα ετικετών:", labels[:10])

def tokenize(text):
    return re.findall(r'\w+', text.lower())

print("Δημιουργία λεξικού...")
word_freq = {}
for text in texts:
    for token in tokenize(text):
        word_freq[token] = word_freq.get(token, 0) + 1

sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
sorted_words = sorted_words[:MAX_VOCAB_SIZE-2]

word2idx = {'<PAD>': 0, '<UNK>': 1}
for word, freq in sorted_words:
    word2idx[word] = len(word2idx)
vocab_size = len(word2idx)
print("Μέγεθος λεξικού:", vocab_size)

def text_to_sequence(text, word2idx, max_len=MAX_SEQ_LEN):
    tokens = tokenize(text)
    seq = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    if len(seq) < max_len:
        seq = seq + [word2idx['<PAD>']] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

print("Μετατροπή κειμένων σε ακολουθίες...")
sequences = [text_to_sequence(text, word2idx) for text in texts]
sequences = np.array(sequences)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
print("Αριθμός δειγμάτων - Εκπαίδευση:", len(X_train), "Test:", len(X_test))

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        hidden = hidden.squeeze(0)
        logits = self.fc(hidden)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Εκπαίδευση του μοντέλου...")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    avg_loss = epoch_loss / total
    accuracy = correct / total * 100
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
test_accuracy = correct / total * 100
print(f"\nΤελική ακρίβεια: {test_accuracy:.2f}%")