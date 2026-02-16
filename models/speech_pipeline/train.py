import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
import librosa

from transformers import AutoProcessor, AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate

# ---------------- Config ----------------
BASE = "/content/drive/MyDrive/tess_emotion_assignment"
OUT_DIR  = f"{BASE}/outputs"
CKPT_DIR = f"{BASE}/checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

SPEECH_MODEL = "facebook/wav2vec2-base"
TARGET_SR = 16000
SEED = 42

# Load splits created in text train
train_df = pd.read_csv(f"{OUT_DIR}/train_df.csv")
val_df   = pd.read_csv(f"{OUT_DIR}/val_df.csv")
test_df  = pd.read_csv(f"{OUT_DIR}/test_df.csv")
with open(f"{OUT_DIR}/labels.txt") as f:
    labels = [line.strip() for line in f if line.strip()]

label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

processor = AutoProcessor.from_pretrained(SPEECH_MODEL)

_AUDIO_CACHE = {}
def load_audio(path: str) -> np.ndarray:
    if path in _AUDIO_CACHE:
        return _AUDIO_CACHE[path]
    x, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    _AUDIO_CACHE[path] = x
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

metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, y = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=y)["accuracy"],
        "macro_f1": metric_f1.compute(predictions=preds, references=y, average="macro")["f1"]
    }

speech_model = AutoModelForAudioClassification.from_pretrained(
    SPEECH_MODEL, num_labels=len(labels), id2label=id2label, label2id=label2id
)

FREEZE_ENCODER = True
if FREEZE_ENCODER:
    for p in speech_model.wav2vec2.parameters():
        p.requires_grad = False
    print("✅ Froze wav2vec2 encoder (head-only training).")

speech_args = TrainingArguments(
    output_dir=f"{CKPT_DIR}/speech_wav2vec2",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    learning_rate=1e-3 if FREEZE_ENCODER else 1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    num_train_epochs=8 if FREEZE_ENCODER else 5,
    weight_decay=0.01,
    fp16=False,
    bf16=False,
    remove_unused_columns=False,
    report_to="none",
    logging_steps=50,
    seed=SEED,
)

trainer = Trainer(
    model=speech_model,
    args=speech_args,
    train_dataset=SpeechDS(train_df),
    eval_dataset=SpeechDS(val_df),
    data_collator=collate_speech,
    compute_metrics=compute_metrics
)

trainer.train()

# ---------------- Extract audio embeddings ----------------
from transformers import AutoModel
device = "cuda" if torch.cuda.is_available() else "cpu"
speech_encoder = AutoModel.from_pretrained(SPEECH_MODEL).to(device).eval()

@torch.no_grad()
def extract_audio_embeddings(df_, batch_size=8):
    paths = df_["audio_path"].tolist()
    y = df_["label_id"].to_numpy()
    embs = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        audios = [load_audio(p) for p in batch_paths]
        inp = processor(audios, sampling_rate=TARGET_SR, return_tensors="pt", padding=True).to(device)
        out = speech_encoder(**inp)
        pooled = out.last_hidden_state.mean(dim=1)
        embs.append(pooled.cpu().numpy())
    return np.vstack(embs), y

audio_train, y_train = extract_audio_embeddings(train_df.reset_index(drop=True))
audio_val,   y_val   = extract_audio_embeddings(val_df.reset_index(drop=True))
audio_test,  y_test  = extract_audio_embeddings(test_df.reset_index(drop=True))

np.save(f"{OUT_DIR}/audio_train.npy", audio_train)
np.save(f"{OUT_DIR}/audio_val.npy",   audio_val)
np.save(f"{OUT_DIR}/audio_test.npy",  audio_test)

print("✅ Speech training done. Saved embeddings to:", OUT_DIR)