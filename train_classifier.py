"""
Ticket Category & Urgency Classification — Training Pipeline
=============================================================
Fine-tunes DistilBERT (base-uncased) with two classification heads:
  • Category  → 6 classes (Refund, Login, Delivery, Billing, Account, Other)
  • Urgency   → 3 classes (High, Medium, Low)

Usage:
    python train_classifier.py

Output:
    ./models/ticket_classifier/
        ├── model.pt          (state dict)
        ├── label_maps.json   (category ↔ id, urgency ↔ id)
        └── tokenizer files   (saved via save_pretrained)
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizerFast

# ──────────────────────────────────────────────
# 0.  REPRODUCIBILITY
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ──────────────────────────────────────────────
# 1.  CONFIGURATION
# ──────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
WEIGHT_DECAY = 0.01
DROPOUT      = 0.3
VAL_RATIO    = 0.2

CATEGORIES = ["Incident", "Request", "Problem", "Change"]
URGENCIES  = ["High", "Medium", "Low"]

DATA_PATH   = "customer_support_tickets.csv"
SAVE_DIR    = "./models/ticket_classifier"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"🖥  Device: {DEVICE}")

# ──────────────────────────────────────────────
# 2.  DATA LOADING / MOCK DATA GENERATION
# ──────────────────────────────────────────────

MOCK_TEXTS = {
    "Refund": [
        "I want a refund for my order, the product was damaged.",
        "Please process a refund, I received the wrong item.",
        "How can I get my money back for a defective product?",
        "I need to return this and get a full refund immediately.",
    ],
    "Login": [
        "I cannot log in to my account, it says invalid password.",
        "My login credentials are not working since yesterday.",
        "I forgot my password and the reset link is not arriving.",
        "Account locked after too many failed login attempts.",
    ],
    "Delivery": [
        "My package has not arrived yet, it has been 10 days.",
        "The delivery status shows stuck in transit for a week.",
        "I received someone else's package instead of mine.",
        "Can I change the delivery address for my current order?",
    ],
    "Billing": [
        "I was charged twice for the same subscription.",
        "There is an unauthorized charge on my credit card.",
        "My invoice amount does not match the agreed price.",
        "Please update my billing information to a new card.",
    ],
    "Account": [
        "I want to delete my account and all associated data.",
        "How do I change my registered email address?",
        "Please help me update my phone number on my profile.",
        "I need to merge two duplicate accounts into one.",
    ],
    "Other": [
        "I have a general question about your privacy policy.",
        "Can you tell me about your loyalty rewards programme?",
        "Is there a student discount available for this service?",
        "How do I leave a review for a product I purchased?",
    ],
}

URGENCY_WEIGHTS = {"High": 0.25, "Medium": 0.50, "Low": 0.25}


def generate_mock_data(n_per_category: int = 20) -> pd.DataFrame:
    """Generate a small but balanced mock dataset for pipeline testing."""
    rows = []
    urgencies = list(URGENCY_WEIGHTS.keys())
    probs = list(URGENCY_WEIGHTS.values())
    for cat, templates in MOCK_TEXTS.items():
        for _ in range(n_per_category):
            text = random.choice(templates)
            # Add slight variation so every row isn't identical
            suffix = random.choice(["", " Please help.", " This is urgent.", " Thanks."])
            urgency = random.choices(urgencies, weights=probs, k=1)[0]
            rows.append({
                "ticket_text": text + suffix,
                "category": cat,
                "urgency": urgency,
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"📝 Generated mock dataset with {len(df)} rows")
    return df


def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        required = {"ticket_text", "category", "urgency"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must have columns {required}, found {set(df.columns)}")
        print(f"📂 Loaded {len(df)} rows from {DATA_PATH}")
    else:
        print(f"⚠️  {DATA_PATH} not found — generating mock data for testing.")
        df = generate_mock_data()
    return df


# ──────────────────────────────────────────────
# 3.  LABEL ENCODING
# ──────────────────────────────────────────────
cat2id = {c: i for i, c in enumerate(CATEGORIES)}
id2cat = {i: c for c, i in cat2id.items()}

urg2id = {u: i for i, u in enumerate(URGENCIES)}
id2urg = {i: u for u, i in urg2id.items()}


# ──────────────────────────────────────────────
# 4.  PYTORCH DATASET
# ──────────────────────────────────────────────
class TicketDataset(Dataset):
    """Tokenizes texts and returns input_ids, attention_mask, and both label tensors."""

    def __init__(self, texts, cat_labels, urg_labels, tokenizer, max_len):
        self.texts      = texts
        self.cat_labels = cat_labels
        self.urg_labels = urg_labels
        self.tokenizer  = tokenizer
        self.max_len    = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),   # (max_len,)
            "cat_label":      torch.tensor(self.cat_labels[idx], dtype=torch.long),  # scalar
            "urg_label":      torch.tensor(self.urg_labels[idx], dtype=torch.long),  # scalar
        }


# ──────────────────────────────────────────────
# 5.  MODEL — Multi-Head DistilBERT
# ──────────────────────────────────────────────
class MultiHeadDistilBERT(nn.Module):
    """
    DistilBERT backbone with two independent classification heads.

    Architecture
    ────────────
    DistilBERT (base-uncased)
        │
        ▼  [CLS] token hidden state  →  (batch, 768)
        ├──▶  Dropout → Linear(768, n_categories)  →  category_logits   (batch, 6)
        └──▶  Dropout → Linear(768, n_urgencies)   →  urgency_logits    (batch, 3)
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        n_categories: int = len(CATEGORIES),
        n_urgencies: int = len(URGENCIES),
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)

        hidden_size = self.distilbert.config.hidden_size  # 768

        # Category classification head
        self.cat_dropout = nn.Dropout(dropout)
        self.cat_head    = nn.Linear(hidden_size, n_categories)

        # Urgency classification head
        self.urg_dropout = nn.Dropout(dropout)
        self.urg_head    = nn.Linear(hidden_size, n_urgencies)

    def forward(self, input_ids, attention_mask):
        """
        Parameters
        ----------
        input_ids      : (batch, seq_len)
        attention_mask  : (batch, seq_len)

        Returns
        -------
        category_logits : (batch, n_categories)   — raw scores for 6 categories
        urgency_logits  : (batch, n_urgencies)    — raw scores for 3 urgencies
        """
        # DistilBERT output: last_hidden_state → (batch, seq_len, hidden_size)
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # (batch, 768) — [CLS] token

        category_logits = self.cat_head(self.cat_dropout(cls_hidden))  # (batch, 6)
        urgency_logits  = self.urg_head(self.urg_dropout(cls_hidden))  # (batch, 3)

        return category_logits, urgency_logits


# ──────────────────────────────────────────────
# 6.  TRAINING UTILITIES
# ──────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, cat_criterion, urg_criterion, device):
    model.train()
    total_loss = 0.0
    all_cat_preds, all_cat_labels = [], []
    all_urg_preds, all_urg_labels = [], []

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cat_labels     = batch["cat_label"].to(device)
        urg_labels     = batch["urg_label"].to(device)

        optimizer.zero_grad()

        cat_logits, urg_logits = model(input_ids, attention_mask)

        # Combined loss (equal weighting — adjust if one task is more important)
        loss_cat = cat_criterion(cat_logits, cat_labels)
        loss_urg = urg_criterion(urg_logits, urg_labels)
        loss = loss_cat + loss_urg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        all_cat_preds.extend(cat_logits.argmax(dim=1).cpu().tolist())
        all_cat_labels.extend(cat_labels.cpu().tolist())
        all_urg_preds.extend(urg_logits.argmax(dim=1).cpu().tolist())
        all_urg_labels.extend(urg_labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    cat_acc  = accuracy_score(all_cat_labels, all_cat_preds)
    urg_acc  = accuracy_score(all_urg_labels, all_urg_preds)
    return avg_loss, cat_acc, urg_acc


@torch.no_grad()
def evaluate(model, dataloader, cat_criterion, urg_criterion, device):
    model.eval()
    total_loss = 0.0
    all_cat_preds, all_cat_labels = [], []
    all_urg_preds, all_urg_labels = [], []

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cat_labels     = batch["cat_label"].to(device)
        urg_labels     = batch["urg_label"].to(device)

        cat_logits, urg_logits = model(input_ids, attention_mask)

        loss_cat = cat_criterion(cat_logits, cat_labels)
        loss_urg = urg_criterion(urg_logits, urg_labels)
        total_loss += (loss_cat + loss_urg).item()

        all_cat_preds.extend(cat_logits.argmax(dim=1).cpu().tolist())
        all_cat_labels.extend(cat_labels.cpu().tolist())
        all_urg_preds.extend(urg_logits.argmax(dim=1).cpu().tolist())
        all_urg_labels.extend(urg_labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    cat_acc  = accuracy_score(all_cat_labels, all_cat_preds)
    urg_acc  = accuracy_score(all_urg_labels, all_urg_preds)
    return avg_loss, cat_acc, urg_acc


# ──────────────────────────────────────────────
# 7.  MAIN
# ──────────────────────────────────────────────

def main():
    # ── Load data ───────────────────────────
    df = load_data()

    texts      = df["ticket_text"].tolist()
    cat_labels = df["category"].map(cat2id).tolist()
    urg_labels = df["urgency"].map(urg2id).tolist()

    # ── Train/Val split (stratify on category) ─
    (
        train_texts, val_texts,
        train_cat, val_cat,
        train_urg, val_urg,
    ) = train_test_split(
        texts, cat_labels, urg_labels,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=cat_labels,
    )
    print(f"📊 Train: {len(train_texts)}  |  Val: {len(val_texts)}")

    # ── Tokenizer ──────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = TicketDataset(train_texts, train_cat, train_urg, tokenizer, MAX_LEN)
    val_ds   = TicketDataset(val_texts, val_cat, val_urg, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ──────────────────────────────
    model = MultiHeadDistilBERT().to(DEVICE)

    # ── Optimizer & Loss ───────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cat_criterion = nn.CrossEntropyLoss()
    urg_criterion = nn.CrossEntropyLoss()

    # ── Training Loop ──────────────────────
    print("\n🚀 Starting training …\n")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_cat_acc, train_urg_acc = train_one_epoch(
            model, train_loader, optimizer, cat_criterion, urg_criterion, DEVICE,
        )
        val_loss, val_cat_acc, val_urg_acc = evaluate(
            model, val_loader, cat_criterion, urg_criterion, DEVICE,
        )

        print(
            f"Epoch {epoch}/{EPOCHS}  │  "
            f"Train Loss: {train_loss:.4f}  Cat Acc: {train_cat_acc:.3f}  Urg Acc: {train_urg_acc:.3f}  │  "
            f"Val Loss: {val_loss:.4f}  Cat Acc: {val_cat_acc:.3f}  Urg Acc: {val_urg_acc:.3f}"
        )

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    # ── Save everything ────────────────────
    save_path = Path(SAVE_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    # Model weights
    torch.save(best_state, save_path / "model.pt")
    print(f"\n💾 Model weights saved  → {save_path / 'model.pt'}")

    # Tokenizer
    tokenizer.save_pretrained(str(save_path))
    print(f"💾 Tokenizer saved      → {save_path}")

    # Label maps
    label_maps = {
        "cat2id": cat2id,
        "id2cat": id2cat,
        "urg2id": urg2id,
        "id2urg": id2urg,
    }
    with open(save_path / "label_maps.json", "w") as f:
        json.dump(label_maps, f, indent=2)
    print(f"💾 Label maps saved     → {save_path / 'label_maps.json'}")

    print("\n✅ Training complete!  Load the model in your FastAPI backend like this:\n")
    print(
        "    model = MultiHeadDistilBERT()\n"
        "    model.load_state_dict(torch.load('./models/ticket_classifier/model.pt'))\n"
        "    model.eval()\n"
        "    tokenizer = DistilBertTokenizerFast.from_pretrained('./models/ticket_classifier/')\n"
    )


if __name__ == "__main__":
    main()
