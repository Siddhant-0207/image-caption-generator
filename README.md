## 🖼️ Image Caption Generator

An end-to-end Deep Learning based Image Caption Generator that takes an image as input and generates a meaningful natural language caption using a CNN + LSTM architecture.

This project uses the Flickr8k dataset, feature extraction with InceptionV3, and sequence modeling with LSTM, deployed via a Flask web interface.

##  🚀 Project Overview

The model follows the standard Encoder–Decoder architecture:

Encoder (CNN – InceptionV3) → Extracts image features

Decoder (LSTM) → Generates captions word by word

Tokenizer → Converts words into numerical sequences

Flask Web App → Allows users to upload images and get captions

## 📂 Project Structure

```bash
Image-Caption-Generator/
│
├── data/
│   ├── Flickr8k_Dataset/
│   ├── Flickr8k_text/
│   └── processed/
│
├── src/
│   ├── model/
│   │   └── caption_model.keras
│   ├── inference/
│   │   └── generate_caption.py
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── preprocess.py
├── tokenize_caption.py
├── extract_feature.py
├── create_sequence.py
├── train_model.py
├── app.py
└── README.md
```
## 📊 Dataset

Dataset Used: Flickr8k

8,000 images

5 captions per image

Total ~40,000 captions

## ⚙️ Installation
### 1️⃣ Clone Repository
git clone https://github.com/Siddhant-0207/image-caption-generator.git
cd image-caption-generator
### 2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
###  Install Dependencies
pip install -r requirements.txt

## Required Libraries:

#### tensorflow
#### numpy
#### tqdm
#### flask
#### pillow

## 🔄 Workflow Pipeline
### 1️⃣ Preprocess Captions

File: preprocess.py

Load raw captions

Clean text (lowercase, remove punctuation, remove numbers)

Build vocabulary

Save cleaned captions & vocab

python preprocess.py

Output:

captions_cleaned.pkl

vocab.txt

### 2️⃣ Tokenization

File: tokenize_caption.py

Add <start> and <end> tokens

Fit Keras Tokenizer

Save tokenizer

Compute max caption length

python tokenize_caption.py

Output:

tokenizer.pkl

### 3️⃣ Extract Image Features

File: extract_feature.py

Load InceptionV3 (pretrained on ImageNet)

Remove final classification layer

Extract 2048-d feature vectors

python extract_feature.py

Output:

image_features.pkl

### 4️⃣ Create Training Sequences

File: create_sequence.py

Generate (image_features, input_sequence) → next_word

Uses data generator for memory efficiency

python create_sequence.py
### 5️⃣ Train Model

File: train_model.py

Model Architecture:

Image Features (2048)
        ↓
      Dense
        ↓
     + Merge +
        ↓
       LSTM
        ↓
     Softmax

Train:

python train_model.py

Output:

caption_model.keras

## 🧠 Model Architecture Details
🔹 Image Branch

Input: 2048-d vector

Dropout

Dense (256 units)

🔹 Text Branch

Embedding Layer

Dropout

LSTM (256 units)

🔹 Decoder

Add() merge

Dense

Softmax output

Loss Function:

Categorical Crossentropy

Optimizer:

Adam (lr=0.001)
## 🌐 Web Application

File: app.py

Built with Flask.

Features:

Upload image

Preview image

Generate caption

Animated UI with Bootstrap

Toast notifications

Loading spinner

Run App:

python app.py

Open in browser:

http://127.0.0.1:5000/
## 🖥️ UI Preview

Dark gradient theme

Animated background elements

Responsive layout

Clean modern card-based design

## 📈 Future Improvements

Use Attention Mechanism

Replace LSTM with Transformer

Use BLEU score for evaluation

Deploy on Render / AWS / HuggingFace

Convert to FastAPI for production

## 🎯 Results

The model successfully generates meaningful captions such as:

"A dog running through the grass"<br>
"A group of people playing football"<br>
"A child jumping into a pool"<br>
