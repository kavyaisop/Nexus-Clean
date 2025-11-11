import os
import random
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -------------------------
# CONFIG
# -------------------------
DATA_FILE = "dataset1_augmented.csv"
MODEL_NAME = "xlm-roberta-base"  # you can switch to 'xlm-roberta-small' if needed
OUTPUT_DIR = "./xlmr-save"
MAX_LENGTH = 64          # shorter for memory efficiency
EPOCHS = 5
BATCH_SIZE = 8          # very small to avoid GPU memory issues
LR = 2e-5
SEED = 42
GRADIENT_ACCUMULATION_STEPS = 6
USE_AMP = True           # use mixed precision if GPU

# -------------------------
# DEVICE & SEED
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) if device.type == "cuda" else None

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv(DATA_FILE)
if not {"text", "label"}.issubset(df.columns):
    raise ValueError("CSV must have 'text' and 'label' columns")
df["label"] = df["label"].astype(int)

train_df = df.sample(frac=0.8, random_state=SEED)
test_df = df.drop(train_df.index)

# -------------------------
# Tokenizer + Encoding
# -------------------------
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

def encode_texts(texts, labels):
    encodings = tokenizer.batch_encode_plus(
        texts.tolist(),
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(labels.tolist(), dtype=torch.long),
    )

train_dataset = encode_texts(train_df["text"], train_df["label"])
test_dataset = encode_texts(test_df["text"], test_df["label"])

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=BATCH_SIZE,
)
test_dataloader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=BATCH_SIZE,
)

# -------------------------
# Model
# -------------------------
num_labels = df["label"].nunique()
model = XLMRobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
)
model.to(device)

# -------------------------
# Optimizer & Scheduler
# -------------------------
optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)
total_steps = len(train_dataloader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)
loss_fn = CrossEntropyLoss()

# -------------------------
# Metrics
# -------------------------
def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }

# -------------------------
# Mixed Precision Scaler
# -------------------------
scaler = torch.cuda.amp.GradScaler() if USE_AMP and device.type == "cuda" else None

# -------------------------
# Training Loop
# -------------------------
best_f1 = 0.0
global_step = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                outputs = model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                loss = outputs[0] / GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss).backward()
        else:
            outputs = model(
                b_input_ids,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            loss = outputs[0] / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

        total_loss += loss.item()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            model.zero_grad()
            global_step += 1

        if step % 5 == 0:
            print(f"Step {step}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"  Average training loss: {avg_train_loss:.4f}")

    # -------------------------
    # Evaluation
    # -------------------------
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            outputs = model(
                b_input_ids,
                attention_mask=b_input_mask,
            )
            logits = outputs[0]
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())

    metrics = compute_metrics(np.array(preds), np.array(true_labels))
    print("📊 Evaluation:", metrics)

    # Save best model
    if metrics["f1_macro"] > best_f1:
        best_f1 = metrics["f1_macro"]
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        model.save_pretrained(os.path.join(OUTPUT_DIR, "best"))
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best"))
        print(f"✅ New best model saved with F1_macro={best_f1:.4f}")

print("\nTraining complete!")
