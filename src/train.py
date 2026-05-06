# src/train.py
# Run from project root: python src/train.py

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ← add this line
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

from src.utils import load_feature_dataset, make_splits, FEATURE_COLS
from src.model import MatchPredictor

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("models", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

df = load_feature_dataset("data/features/feature_dataset.csv")
X_train, X_val, X_test, y_train, y_val, y_test, df_test = make_splits(df)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE CLASS WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
# Because home wins (class 2) are 48% of data, the model might lazily predict
# "home win" for everything and still get 48% accuracy. Class weights fix this
# by penalizing mistakes on rare classes more heavily during training.
#
# compute_class_weight returns weights inversely proportional to class frequency:
#   rarer class → higher weight → model pays more attention to it.

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=y_train
)
print(f"\nClass weights: away={class_weights[0]:.3f}, draw={class_weights[1]:.3f}, home={class_weights[2]:.3f}")
# Expected: draw will have the highest weight (it's the least frequent)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BASELINE: RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BASELINE MODEL: Random Forest")
print("="*60)

# Why Random Forest?
#   It's an ensemble of decision trees. Each tree votes on the outcome, and
#   the majority vote wins. It handles tabular data well, requires almost no
#   tuning, and gives a solid baseline to compare the neural network against.
#
# class_weight="balanced": tells sklearn to apply the same weighting logic
#   as we computed above — built directly into the model.
#
# n_estimators=200: use 200 trees. More trees = more stable predictions,
#   but slower training. 200 is a good balance for this dataset size.

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",  # handles class imbalance
    random_state=SEED,
    n_jobs=-1,                # use all CPU cores
)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

# Evaluate on validation set
rf_val_preds = rf_model.predict(X_val)
rf_val_acc = accuracy_score(y_val, rf_val_preds)
print(f"\nValidation Accuracy: {rf_val_acc:.4f} ({rf_val_acc*100:.2f}%)")

# Evaluate on test set (2018 & 2022 WC)
rf_test_preds = rf_model.predict(X_test)
rf_test_acc = accuracy_score(y_test, rf_test_preds)
print(f"Test Accuracy (WC 2018+2022): {rf_test_acc:.4f} ({rf_test_acc*100:.2f}%)")

print("\nClassification Report (Test Set):")
print(classification_report(
    y_test, rf_test_preds,
    target_names=["Away Win (0)", "Draw (1)", "Home Win (2)"]
))

# Confusion matrix
cm_rf = confusion_matrix(y_test, rf_test_preds)
disp = ConfusionMatrixDisplay(cm_rf, display_labels=["Away Win", "Draw", "Home Win"])
fig, ax = plt.subplots(figsize=(7, 5))
disp.plot(ax=ax, colorbar=False)
ax.set_title("Random Forest — Confusion Matrix (WC Test Set)")
plt.tight_layout()
plt.savefig("models/rf_confusion_matrix.png", dpi=120)

print("Saved: models/rf_confusion_matrix.png")

# Feature importance (bonus insight)
importances = rf_model.feature_importances_
feat_series = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 4))
feat_series.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Random Forest — Feature Importances")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("models/rf_feature_importance.png", dpi=120)
plt.show()
print("Saved: models/rf_feature_importance.png")

# Save the model
with open("models/baseline_sklearn.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("Saved: models/baseline_sklearn.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NEURAL NETWORK: PyTorch
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("NEURAL NETWORK: PyTorch")
print("="*60)

# --- 4a. Convert data to PyTorch tensors ---
# Neural networks in PyTorch operate on Tensors, not NumPy arrays.
# float32 is the standard dtype for neural network weights and inputs.
# LongTensor for labels because CrossEntropyLoss expects integer class indices.

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t  = torch.tensor(y_train, dtype=torch.long)
X_val_t    = torch.tensor(X_val,   dtype=torch.float32)
y_val_t    = torch.tensor(y_val,   dtype=torch.long)
X_test_t   = torch.tensor(X_test,  dtype=torch.float32)
y_test_t   = torch.tensor(y_test,  dtype=torch.long)

# --- 4b. DataLoaders ---
# DataLoaders handle batching and shuffling automatically.
# batch_size=256: process 256 matches at a time instead of all at once.
# This makes training faster and the gradient updates less noisy.

BATCH_SIZE = 256

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# --- 4c. Model, Loss, Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MatchPredictor(input_dim=9, dropout_rate=0.3).to(device)
print(f"\nModel architecture:\n{model}")

# CrossEntropyLoss with class weights
# weight= tells the loss to multiply the error for each class by its weight.
# Same effect as class_weight="balanced" in sklearn.
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# Adam optimizer
# Why Adam?
#   Adam (Adaptive Moment Estimation) automatically adjusts the learning rate
#   for each parameter based on recent gradients. It converges faster and
#   is more forgiving of hyperparameter choices than vanilla SGD.
#   lr=1e-3 (0.001) is the standard starting point for Adam.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Learning rate scheduler: reduce LR when validation loss plateaus
# This helps squeeze out extra performance once the model stops improving quickly.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

# --- 4d. Training loop ---
EPOCHS = 60
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_loss = float("inf")

print(f"\nTraining for {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):

    # ── Training phase ──────────────────────────────────────────
    model.train()   # enables dropout
    epoch_train_loss = 0.0
    correct_train = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()           # clear gradients from last step
        logits = model(X_batch)         # forward pass
        loss = criterion(logits, y_batch)  # compute loss
        loss.backward()                 # backpropagation
        optimizer.step()                # update weights

        epoch_train_loss += loss.item() * len(y_batch)
        correct_train += (logits.argmax(dim=1) == y_batch).sum().item()

    avg_train_loss = epoch_train_loss / len(y_train)
    train_acc = correct_train / len(y_train)

    # ── Validation phase ─────────────────────────────────────────
    model.eval()    # disables dropout — we want deterministic predictions
    epoch_val_loss = 0.0
    correct_val = 0

    with torch.no_grad():   # no gradient computation needed for evaluation
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            epoch_val_loss += loss.item() * len(y_batch)
            correct_val += (logits.argmax(dim=1) == y_batch).sum().item()

    avg_val_loss = epoch_val_loss / len(y_val)
    val_acc = correct_val / len(y_val)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Step the scheduler based on validation loss
    scheduler.step(avg_val_loss)

    # Save the best model checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "models/neural_net_pytorch.pth")

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

print(f"\nBest validation loss: {best_val_loss:.4f}")
print("Saved: models/neural_net_pytorch.pth  (best checkpoint)")

# --- 4e. Loss & accuracy curves ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(train_losses, label="Train Loss", color="steelblue")
axes[0].plot(val_losses,   label="Val Loss",   color="coral")
axes[0].set_title("Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("CrossEntropy Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot([a*100 for a in train_accs], label="Train Acc", color="steelblue")
axes[1].plot([a*100 for a in val_accs],   label="Val Acc",   color="coral")
axes[1].set_title("Accuracy Curves")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/nn_training_curves.png", dpi=120)
plt.show()
print("Saved: models/nn_training_curves.png")

# --- 4f. Test set evaluation ---
# Load the best checkpoint (not just the last epoch's weights)
model.load_state_dict(torch.load("models/neural_net_pytorch.pth", map_location=device))
model.eval()

with torch.no_grad():
    nn_test_logits = model(X_test_t.to(device))
    nn_test_preds = nn_test_logits.argmax(dim=1).cpu().numpy()

nn_test_acc = accuracy_score(y_test, nn_test_preds)
print(f"\nNeural Network Test Accuracy (WC 2018+2022): {nn_test_acc:.4f} ({nn_test_acc*100:.2f}%)")

print("\nClassification Report (Test Set):")
print(classification_report(
    y_test, nn_test_preds,
    target_names=["Away Win (0)", "Draw (1)", "Home Win (2)"]
))

# Confusion matrix
cm_nn = confusion_matrix(y_test, nn_test_preds)
disp_nn = ConfusionMatrixDisplay(cm_nn, display_labels=["Away Win", "Draw", "Home Win"])
fig, ax = plt.subplots(figsize=(7, 5))
disp_nn.plot(ax=ax, colorbar=False)
ax.set_title("Neural Network — Confusion Matrix (WC Test Set)")
plt.tight_layout()
plt.savefig("models/nn_confusion_matrix.png", dpi=120)
plt.show()
print("Saved: models/nn_confusion_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HEAD-TO-HEAD COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<30} {'Test Accuracy':>15}")
print("-" * 46)
print(f"{'Random Forest (baseline)':<30} {rf_test_acc*100:>14.2f}%")
print(f"{'Neural Network (PyTorch)':<30} {nn_test_acc*100:>14.2f}%")
print("="*60)
print("\nPhase 3 complete. Both models saved to models/")