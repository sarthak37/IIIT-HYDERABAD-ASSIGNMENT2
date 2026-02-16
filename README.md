# IIIT-H Assignment 2 — TESS Emotion Classification (Text, Speech, Fusion)

This project implements three variants for emotion classification on the **TESS** dataset:
1. **Text-only (Contextual Modelling)**: DeBERTa-v3-base fine-tuned for emotion classification  
2. **Speech-only (Temporal Modelling)**: Wav2Vec2-base (encoder frozen, classification head trained)  
3. **Multimodal Fusion**: Late fusion by concatenating text + speech embeddings and training an MLP

OUTPUT SCREENSHOTS:-

<img width="1600" height="1200" alt="umap_text" src="https://github.com/user-attachments/assets/11fda7b0-3c33-484c-a318-920ce32c9d47" />

<img width="1600" height="1200" alt="umap_speech" src="https://github.com/user-attachments/assets/7da77716-a671-4dc9-88b9-c9de7f1f551a" />

<img width="1600" height="1200" alt="umap_fusion" src="https://github.com/user-attachments/assets/f65bc67b-276a-496f-bc12-2ace3583012b" />

<img width="758" height="747" alt="Text_confusion_matrix" src="https://github.com/user-attachments/assets/0473eea0-c881-4179-be27-a70a4f970757" />

<img width="723" height="630" alt="Speech_confusion_matrix" src="https://github.com/user-attachments/assets/cb6e95ee-a6a8-4a67-a887-f76c04d45257" />

<img width="750" height="629" alt="Fusion_confusion_matrix" src="https://github.com/user-attachments/assets/5f616fe5-3d3f-446f-826b-3164a5d8584b" />
