import os, glob, re, pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

# ---------------- Config ----------------
BASE = "/content/drive/MyDrive/tess_emotion_assignment"
DATA_DIR = f"{BASE}/data"
OUT_DIR  = f"{BASE}/outputs"
CKPT_DIR = f"{BASE}/checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

TEXT_MODEL = "microsoft/deberta-v3-base"
SEED = 42

# ---------------- Data build ----------------
wav_files = glob.glob(f"{DATA_DIR}/**/*.wav", recursive=True)
assert len(wav_files) > 0, f"No .wav found under {DATA_DIR}. Unzip dataset there."

def infer_emotion_from_path(p: str):
    low = p.lower()
    if "pleasant_surprise" in low or re.search(r"[_-]ps[_-]", low):
        return "pleasant_surprise"
    for e in ["angry","disgust","fear","happy","neutral","sad","surprise"]:
        if re.search(rf"[/_ -]{e}([/_ -]|$)", low) or f"_{e}." in low:
            return e
    folder = pathlib.Path(p).parent.name.lower()
    if "pleasant" in folder and "surprise" in folder:
        return "pleasant_surprise"
    return None

def infer_transcript_from_filename(p: str):
    name = pathlib.Path(p).stem.lower()
    name = re.sub(r"^(oaf|yaf)_", "", name)
    name = re.sub(r"_(angry|disgust|fear|happy|neutral|sad|surprise|pleasant_surprise|ps)$", "", name)
    return name.replace("_", " ").strip() or "unknown"

rows = []
for p in wav_files:
    emo = infer_emotion_from_path(p)
    if emo is None:
        continue
    rows.append({"audio_path": p, "text": infer_transcript_from_filename(p), "label": emo})

df = pd.DataFrame(rows)
labels = sorted(df["label"].unique().tolist())
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["label_id"])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label_id"])

# Save splits (optional but helpful)
df.to_csv(f"{OUT_DIR}/full_df.csv", index=False)
train_df.to_csv(f"{OUT_DIR}/train_df.csv", index=False)
val_df.to_csv(f"{OUT_DIR}/val_df.csv", index=False)
test_df.to_csv(f"{OUT_DIR}/test_df.csv", index=False)

# ---------------- Dataset ----------------
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

# ---------------- Metrics ----------------
metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, y = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=y)["accuracy"],
        "macro_f1": metric_f1.compute(predictions=preds, references=y, average="macro")["f1"]
    }

# ---------------- Train ----------------
model = AutoModelForSequenceClassification.from_pretrained(
    TEXT_MODEL, num_labels=len(labels), id2label=id2label, label2id=label2id
)

args = TrainingArguments(
    output_dir=f"{CKPT_DIR}/text_deberta",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=False,
    bf16=False,  # keep stable across GPUs
    report_to="none",
    logging_steps=50,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=TextDS(train_df),
    eval_dataset=TextDS(val_df),
    data_collator=collate_text,
    compute_metrics=compute_metrics
)

trainer.train()

# ---------------- Extract embeddings (base encoder) ----------------
from transformers import AutoModel
device = "cuda" if torch.cuda.is_available() else "cpu"
text_encoder = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()

@torch.no_grad()
def extract_text_embeddings(df_, batch_size=64):
    texts = df_["text"].tolist()
    y = df_["label_id"].to_numpy()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        out = text_encoder(**enc)
        cls = out.last_hidden_state[:, 0, :]
        embs.append(cls.cpu().numpy())
    return np.vstack(embs), y

text_train_emb, y_train = extract_text_embeddings(train_df.reset_index(drop=True))
text_val_emb,   y_val   = extract_text_embeddings(val_df.reset_index(drop=True))
text_test_emb,  y_test  = extract_text_embeddings(test_df.reset_index(drop=True))

np.save(f"{OUT_DIR}/text_train.npy", text_train_emb)
np.save(f"{OUT_DIR}/text_val.npy",   text_val_emb)
np.save(f"{OUT_DIR}/text_test.npy",  text_test_emb)

np.save(f"{OUT_DIR}/y_train.npy", y_train)
np.save(f"{OUT_DIR}/y_val.npy",   y_val)
np.save(f"{OUT_DIR}/y_test.npy",  y_test)

# Save labels order
with open(f"{OUT_DIR}/labels.txt", "w") as f:
    for l in labels:
        f.write(l + "\n")

print("✅ Text training done.")
print("Saved embeddings to:", OUT_DIR)