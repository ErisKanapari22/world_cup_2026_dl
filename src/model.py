import torch
import torch.nn as nn


class MatchPredictor(nn.Module):
    """
    Feedforward neural network to predict match outcome (away win / draw / home win).

    Architecture:
        Input (9) → FC(64) → ReLU → Dropout
                  → FC(32) → ReLU → Dropout
                  → FC(16) → ReLU
                  → FC(3)  → (raw logits, no Softmax here)

    Why ReLU?
        ReLU (Rectified Linear Unit) sets negative values to 0 and keeps
        positive values unchanged. It's simple, fast, and avoids the
        "vanishing gradient" problem that older activations like Sigmoid suffer
        from — meaning gradients flow better during training.

    Why Dropout?
        Dropout randomly turns off a fraction of neurons during each training
        step. This forces the network to not rely on any single neuron too
        much, preventing overfitting (memorizing training data instead of
        learning real patterns).

    Why no Softmax in forward()?
        PyTorch's CrossEntropyLoss already applies Softmax internally.
        Adding it again would break the math. We only apply Softmax at
        inference time (when we want actual probabilities).
    """

    def __init__(self, input_dim: int = 9, dropout_rate: float = 0.3):
        super(MatchPredictor, self).__init__()

        self.network = nn.Sequential(
            # --- Layer 1 ---
            nn.Linear(input_dim, 64),   # 9 inputs → 64 neurons
            nn.ReLU(),
            nn.Dropout(dropout_rate),   # randomly zero out 30% of neurons

            # --- Layer 2 ---
            nn.Linear(64, 32),          # 64 → 32 neurons
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # --- Layer 3 ---
            nn.Linear(32, 16),          # 32 → 16 neurons
            nn.ReLU(),

            # --- Output layer ---
            nn.Linear(16, 3),           # 16 → 3 classes (away win, draw, home win)
            # No Softmax here — CrossEntropyLoss handles it
        )

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        """Return Softmax probabilities (use this at inference time)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)