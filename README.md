# Amharic Sentiment Classification with BERT

This project fine-tunes **BERT (bert-base-multilingual-cased)** on the [Amharic Sentiment Dataset](https://huggingface.co/datasets/rasyosef/amharic-sentiment) to classify tweets as **positive** or **negative**.

---

##  Features
- Preprocessing and label mapping (`positive`, `negative`)
- Fine-tuning BERT for sequence classification
- Train/validation split with PyTorch `DataLoader`
- Loss function: Cross-Entropy Loss (with padding ignored)
- Optimizer: Adam
- Custom **prediction function** for inference

---

