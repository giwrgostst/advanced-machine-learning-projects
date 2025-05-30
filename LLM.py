"""
Dataset: WikiText-2
        https://huggingface.co/datasets/wikitext
Tokenizer: SentencePiece BPE (trained on WikiText-2 train split)
"""
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

DATA_ROOT      = os.path.expanduser("~/wikitext-2")
SP_MODEL       = "spm.model"
VOCAB_SIZE     = 16000
SEQ_LEN        = 64
BATCH_SIZE     = 32
GRAD_ACCUM     = 2
EMB_SIZE       = 256
NUM_HEADS      = 8
NUM_LAYERS     = 4
FF_HID_DIM     = 1024
DROPOUT        = 0.1
LR             = 3e-4
EPOCHS         = 6
MAX_GEN_LEN    = 50
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

if not os.path.isfile(SP_MODEL):
    spm.SentencePieceTrainer.Train(
        f"--input={os.path.join(DATA_ROOT,'wiki.train.tokens')} "
        f"--model_prefix=spm --vocab_size={VOCAB_SIZE} "
        "--model_type=bpe --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3"
    )

sp = spm.SentencePieceProcessor()
sp.Load(SP_MODEL)
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0,1,2,3

def load_and_encode(fname):
    text = open(os.path.join(DATA_ROOT, fname), "r", encoding="utf8").read().splitlines()
    ids = []
    for line in text:
        if not line:
            continue
        ids_line = sp.EncodeAsIds(line)
        ids.extend([BOS_ID] + ids_line + [EOS_ID])
    return ids

train_ids = load_and_encode("wiki.train.tokens")
valid_ids = load_and_encode("wiki.valid.tokens")

class LMDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.ids = ids
        self.seq_len = seq_len
    def __len__(self):
        return (len(self.ids) - 1) // self.seq_len
    def __getitem__(self, idx):
        i = idx * self.seq_len
        seq = self.ids[i : i + self.seq_len + 1]
        return torch.tensor(seq, dtype=torch.long)

def collate_batch(batch):
    batch = torch.stack(batch)
    return batch[:, :-1], batch[:, 1:]

train_ds = LMDataset(train_ids, SEQ_LEN)
valid_ds = LMDataset(valid_ids, SEQ_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_batch, pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_batch, pin_memory=True)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.nh = n_heads
        self.qkv = nn.Linear(d_model, d_model*3)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B,T,E = x.size()
        qkv = self.qkv(x).view(B,T,3,self.nh,self.d_k).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1,2).contiguous().view(B,T,E)
        return self.fc_out(self.dropout(out))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:,:x.size(1)]

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff,
                 n_layers, seq_len, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mha": MultiHeadAttention(d_model,n_heads,dropout),
                "ln1": nn.LayerNorm(d_model),
                "ff":  FeedForward(d_model,d_ff,dropout),
                "ln2": nn.LayerNorm(d_model),
            }) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
    def _make_mask(self, T, device):
        mask = torch.tril(torch.ones(T,T,device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)
    def forward(self, x):
        B,T = x.size()
        mask = self._make_mask(T, x.device)
        x = self.tok_emb(x) * math.sqrt(self.tok_emb.embedding_dim)
        x = self.pos_enc(x)
        for lyr in self.layers:
            a = lyr["mha"](x, mask)
            x = lyr["ln1"](x + a)
            f = lyr["ff"](x)
            x = lyr["ln2"](x + f)
        return self.fc_out(x)

model = TransformerLM(
    vocab_size=VOCAB_SIZE,
    d_model=EMB_SIZE,
    n_heads=NUM_HEADS,
    d_ff=FF_HID_DIM,
    n_layers=NUM_LAYERS,
    seq_len=SEQ_LEN,
    dropout=DROPOUT,
).to(DEVICE)

optim = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9,0.98), eps=1e-9)
scaler = GradScaler()
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

def top_k_top_p_filtering(logits, top_k=50, top_p=0.95, filter_value=-1e9):
    logits = logits.clone()
    if top_k > 0:
        val, idx = logits.topk(top_k)
        min_val = val[:, -1].unsqueeze(1)
        logits[logits < min_val] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        cutoff = cum_probs > top_p
        cutoff[...,1:] = cutoff[..., :-1].clone()
        cutoff[...,0] = False
        sorted_logits[cutoff] = filter_value
        inv_idx = sorted_idx.argsort(dim=-1)
        logits = sorted_logits.gather(-1, inv_idx)
    return logits

print("=== Training ===")
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    model.train()
    total_loss = 0.0
    optim.zero_grad()
    for i, (xb, yb) in enumerate(train_loader, 1):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with autocast():
            logits = model(xb)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                yb.view(-1)
            ) / GRAD_ACCUM
        scaler.scale(loss).backward()
        total_loss += loss.item() * GRAD_ACCUM

        if i % GRAD_ACCUM == 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

        if i % 500 == 0:
            avg = total_loss / (i)
            print(f"Epoch {epoch} | Batch {i}/{len(train_loader)} | Avg Loss {avg:.4f}")

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for xb, yb in valid_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            val_loss += criterion(
                logits.view(-1,logits.size(-1)),
                yb.view(-1)
            ).item()
        val_ppl = math.exp(val_loss / len(valid_loader))
    print(f"Epoch {epoch} done in {(time.time()-t0):.1f}s — Train Loss: {total_loss/len(train_loader):.4f} — Val PPL: {val_ppl:.2f}")

def generate(prompt, max_len=MAX_GEN_LEN, top_k=50, top_p=0.9):
    model.eval()
    ids = [BOS_ID] + sp.EncodeAsIds(prompt)
    for _ in range(max_len):
        inp = torch.tensor(ids[-SEQ_LEN:], dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(inp)[0, -1]
        filtered = top_k_top_p_filtering(logits, top_k, top_p)
        probs = F.softmax(filtered, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).item()
        ids.append(nxt)
        if nxt == EOS_ID:
            break
    return sp.DecodeIds(ids)

print("\n--- Chatbot ready! (type 'exit' to quit) ---")
while True:
    q = input(">> ").strip()
    if q.lower() in ("exit", "quit"):
        break
    print(generate(q))
