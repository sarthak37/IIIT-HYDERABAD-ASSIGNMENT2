import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset

# ---------------- Config ----------------
BASE = "/content/drive/MyDrive/tess_emotion_assignment"
OUT_DIR  = f"{BASE}/outputs"
CKPT_DIR = f"{BASE}/checkpoints"
RESULTS_DIR = f"{BASE}/Results"
PLOTS_DIR = f"{RESULTS_DIR}/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

TEXT_MODEL = "microsoft/deberta-v3-base"

# Load splits saved by train.py
test_df = pd.read_csv(f"{OUT_DIR}/test_df.csv")
with open(f"{OUT_DIR}/labels.txt") as f:
    labels = [line.strip() for line in f if line.strip()]

label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

class TextDS(Dataset):
    def __init__(self, df_):
        self.texts = df_["text"].tolist()
        self.labels = df_["label_id"].tolist()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        enc = tokenizer(self.texts[i], truncation=True, max_length=128, return_tensors="pt")
        item = {k:v.squeeze(0) for k,v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item

def collate_text(batch):
    labels_ = torch.stack([b["labels"] for b in batch])
    inputs = [{k:v for k,v in b.items() if k!="labels"} for b in batch]
    padded = tokenizer.pad(inputs, return_tensors="pt")
    padded["labels"] = labels_
    return padded

# Load best checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    f"{CKPT_DIR}/text_deberta",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(output_dir="/tmp/text_test", per_device_eval_batch_size=32, report_to="none")
trainer = Trainer(model=model, args=args, data_collator=collate_text)

pred_out = trainer.predict(TextDS(test_df))
logits = pred_out.predictions
y_true = pred_out.label_ids
y_pred = np.argmax(logits, axis=-1)

acc = accuracy_score(y_true, y_pred)
mf1 = f1_score(y_true, y_pred, average="macro")

print("Text Test Accuracy:", acc)
print("Text Test Macro F1:", mf1)
rep = classification_report(y_true, y_pred, target_names=labels, digits=4)
print(rep)

# Save report
with open(f"{RESULTS_DIR}/report_text.txt", "w") as f:
    f.write(rep)
with open(f"{RESULTS_DIR}/metrics_text.txt", "w") as f:
    f.write(f"accuracy={acc}\nmacro_f1={mf1}\n")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax, xticks_rotation=45, values_format="d", colorbar=False)
ax.set_title("Text-only Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/cm_text.png", dpi=200)
plt.close(fig)

print("✅ Saved:", f"{PLOTS_DIR}/cm_text.png")