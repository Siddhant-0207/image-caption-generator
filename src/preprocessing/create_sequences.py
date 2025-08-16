import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ===== Paths =====
PROCESSED_DIR = os.path.join("data", "processed")
TOKENIZER_PATH = os.path.join(PROCESSED_DIR, "tokenizer.pkl")
CLEANED_CAPTIONS_PATH = os.path.join(PROCESSED_DIR, "captions_cleaned.pkl")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "image_features.pkl")  # pre-extracted CNN features

# ===== Parameters =====
BATCH_SIZE = 64  # Adjust based on GPU/CPU RAM
VOCAB_SIZE = None
MAX_LENGTH = None

# ===== Step 1: Load tokenizer, captions, features =====
print("📂 Loading tokenizer & captions...")
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
VOCAB_SIZE = len(tokenizer.word_index) + 1

with open(CLEANED_CAPTIONS_PATH, 'rb') as f:
    cleaned_captions = pickle.load(f)

with open(FEATURES_PATH, 'rb') as f:
    image_features = pickle.load(f)

print(f"✅ Vocabulary size: {VOCAB_SIZE}")
print(f"✅ Captions for {len(cleaned_captions)} images loaded.")
print(f"✅ Image features loaded for {len(image_features)} images.")

# ===== Step 2: Find max caption length =====
all_captions = []
for caps in cleaned_captions.values():
    for c in caps:
        all_captions.append(f"<start> {c} <end>")
MAX_LENGTH = max(len(seq) for seq in tokenizer.texts_to_sequences(all_captions))
print(f"📏 Max caption length: {MAX_LENGTH}")

# ===== Step 3: Define data generator =====
def data_generator(captions, features, tokenizer, max_length, vocab_size, batch_size):
    """Yields ([image_features, input_sequence], output_word) for model training"""
    while True:
        X1, X2, y = [], [], []
        for img_id, caps in captions.items():
            feature = features[img_id]
            for c in caps:
                seq = tokenizer.texts_to_sequences([f"<start> {c} <end>"])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)

                    if len(X1) == batch_size:
                        yield (np.array(X1), np.array(X2)), np.array(y)

                        X1, X2, y = [], [], []

# ===== Step 4: Example usage =====
if __name__ == "__main__":
    # Just to test the generator
    gen = data_generator(cleaned_captions, image_features, tokenizer, MAX_LENGTH, VOCAB_SIZE, BATCH_SIZE)
    X, y = next(gen)
    print(f"🔹 Batch shapes -> Image features: {X[0].shape}, Sequences: {X[1].shape}, Output: {y.shape}")
    print("✅ Generator works! Ready for training.")
