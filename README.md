# Tweet-Emotion-Detection
Fine-tuning transformer models using **PEFT** and **prompt-based learning** for multi-label sentiment classification of tweets.

---

## 📌 Project Overview

This project explores the evolution of tweet emotion detection using various deep learning strategies, progressively optimizing for **accuracy**, **efficiency**, and **cost**.

### 🧪 Stages of Development

- **HW5**: Custom MLP Model (Baseline)
- **HW6**: Full Fine-Tuning with Transformers (RoBERTa, ALBERT, DistilBERT)
- **HW7**: Parameter-Efficient Fine-Tuning (PEFT) using LoRA & BitsAndBytes
- **HW8**: Prompt-Based Fine-Tuning with PEFT (Meta LLaMA-3.2-1B-Instruct)

> 🚀 **Final Model**: `Meta LLaMA-3.2-1B-Instruct` with PEFT + Prompt-Based Tuning  
> 🎯 **Accuracy**: 51.58% in detecting *optimism*

---

## 🛠 Tech Stack

- **Language**: Python  
- **Libraries**: `torch`, `transformers`, `peft`, `datasets`, `accelerate`, `wandb`  
- **Models**: RoBERTa, ALBERT, DistilBERT, LLaMA-3.2-1B, E5-Mistral-7B, Gemma  

---

## 🔧 Installation

Ensure you have Python 3.8+ installed, then install required dependencies:

pip install torch transformers peft datasets accelerate wand

## 📂 Project Structure

tweet-emotion-detection/
├── HW5/                   # Custom MLP Model
├── HW6/                   # Transformer Fine-Tuning
├── HW7/                   # PEFT-Based Fine-Tuning
├── HW8/                   # Prompt-Based Tuning with PEFT
├── results/               # Submission Files & Evaluation Outputs
├── models/                # Saved Fine-Tuned Models (optional)
├── train.py               # Training Script
├── evaluate.py            # Evaluation Script
└── README.md              # Project Documentation
