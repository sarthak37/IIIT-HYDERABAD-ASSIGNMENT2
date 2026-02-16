import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

# ---------------- Config ----------------
BASE = "/content/drive/MyDrive/tess_emotion_assignment"
OUT_DIR  = f"{BASE}/outputs"
CKPT_DIR = f"{BASE}/checkpoints"
os.makedirs(f"{CKPT_DIR}/fusion", exist_ok=True)

with open(f"{OUT_DIR}/labels.txt") as f:
    labels = [line.strip() for line in f if line.strip()]

def load_npy(name):
    path = os.path.join(OUT_DIR, name)
    assert os.path.exists(path), f"Missing {path}"
    return np.load(path)

text_train = load_npy("text_train.npy")
text_val   = load_npy("text_val.npy")

audio_train = load_npy("audio_train.npy")
audio_val   = load_npy("audio_val.npy")

y_train = load_npy("y_train.npy")
y_val   = load_npy("y_val.npy")

X_train = np.concatenate([audio_train, text_train], axis=1)
X_val   = np.concatenate([audio_val,   text_val], axis=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

Xtr = torch.tensor(X_train, dtype=torch.float32)
Ytr = torch.tensor(y_train, dtype=torch.long)
Xva = torch.tensor(X_val, dtype=torch.float32)
Yva = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=128, shuffle=False)

class FusionMLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes)
        )
    def forward(self, x):
        return self.net(x)

model = FusionMLP(Xtr.shape[1], len(labels)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

best_f1 = -1.0
best_state = None

for epoch in range(1, 21):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb.to(device)).argmax(dim=-1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb.numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    val_acc = accuracy_score(all_true, all_pred)
    val_f1  = f1_score(all_true, all_pred, average="macro")

    print(f"Epoch {epoch:02d} | val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
torch.save(model.state_dict(), f"{CKPT_DIR}/fusion/fusion.pt")
print("✅ Saved fusion model:", f"{CKPT_DIR}/fusion/fusion.pt")