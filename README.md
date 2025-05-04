# 🧠 Grammar Scoring Engine

This project builds an AI system to evaluate grammar proficiency from spoken English audio. It predicts a continuous grammar score (0–5) based on Mean Opinion Score (MOS) standards using deep audio embeddings.

## 🔍 Task
Given 45–60 sec `.wav` files, predict grammar accuracy scores based on sentence structure, syntax control, and fluency.

## 📁 Dataset
- `train.csv` – 444 labeled audio samples
- `test.csv` – 195 unlabeled audio samples
- `audios/train/` and `audios/test/` – corresponding `.wav` files

## ⚙️ Approach
1. **Baseline**: Wav2Vec2 embeddings + LightGBM  
2. **Final Model**: HuBERT (frozen) + custom regression MLP head

## 🧹 Preprocessing
- Mono conversion, silence trimming, normalization
- Resampling to 16kHz
- Padding using PyTorch's `pad_sequence`

## 🏋️ Training
- Loss: MSELoss  
- Optimizer: Adam  
- Batch size: 1 (due to sequence length & model size)  
- Epochs: 5  

## 📤 Output
Generates `submission.csv` with predicted grammar scores for test audio.

## 🙏 Credits
- Towards Data Science
- ChatGPT (Free Version)
- Perplexity AI
