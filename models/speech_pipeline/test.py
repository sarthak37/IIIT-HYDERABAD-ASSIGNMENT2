import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import librosa
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForAudioClassification, TrainingArguments, Trainer

# ---------------- Config ----------------
BASE = "/content/drive/MyDrive/tess_emotion_assignment"
OUT_DIR  = f"{BASE}/outputs"
CKPT_DIR = f"{BASE}/checkpoints"
RESULTS_DIR = f"{BASE}/Results"
PLOTS_DIR = f"{RESULTS_DIR}/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

SPEECH_MODEL = "facebook/wav2vec2-base"
TARGET_SR = 16000

test_df = pd.read_csv(f"{OUT_DIR}/test_df.csv")
with open(f"{OUT_DIR}/labels.txt") as f:
    labels = [line.strip() for line in f if line.strip()]

label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

processor = AutoProcessor.from_pretrained(SPEECH_MODEL)

def load_audio(path: str) -> np.ndarray:
    x, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return x

class SpeechDS(Dataset):
    def __init__(self, df_):
        self.paths = df_["audio_path"].tolist()
        self.labels = df_["label_id"].tolist()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"audio": load_audio(self.paths[i]), "labels": int(self.labels[i])}

def collate_speech(batch):
    audios = [b["audio"] for b in batch]
    labels_ = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    inputs = processor(audios, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs["labels"] = labels_
    return inputs

model = AutoModelForAudioClassification.from_pretrained(
    f"{CKPT_DIR}/speech_wav2vec2",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(output_dir="/tmp/speech_test", per_device_eval_batch_size=4, report_to="none", remove_unused_columns=False)
trainer = Trainer(model=model, args=args, data_collator=collate_speech)

pred = trainer.predict(SpeechDS(test_df))
logits = pred.predictions
y_true = pred.label_ids
y_pred = np.argmax(logits, axis=-1)

acc = accuracy_score(y_true, y_pred)
mf1 = f1_score(y_true, y_pred, average="macro")

print("Speech Test Accuracy:", acc)
print("Speech Test Macro F1:", mf1)
rep = classification_report(y_true, y_pred, target_names=labels, digits=4)
print(rep)

with open(f"{RESULTS_DIR}/report_speech.txt", "w") as f:
    f.write(rep)
with open(f"{RESULTS_DIR}/metrics_speech.txt", "w") as f:
    f.write(f"accuracy={acc}\nmacro_f1={mf1}\n")

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax, xticks_rotation=45, values_format="d", colorbar=False)
ax.set_title("Speech-only Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/cm_speech.png", dpi=200)
plt.close(fig)

print("✅ Saved:", f"{PLOTS_DIR}/cm_speech.png")