import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===== Paths =====
PROCESSED_DIR = os.path.join("data", "processed")
CLEANED_CAPTIONS_PATH = os.path.join(PROCESSED_DIR, "captions_cleaned.pkl")
TOKENIZER_PATH = os.path.join(PROCESSED_DIR, "tokenizer.pkl")

# ===== Parameters =====
VOCAB_SIZE = None   # None = use all words
MAX_LENGTH = 0      # will calculate from data

# ===== Step 1: Load cleaned captions =====
print("📂 Loading cleaned captions...")
with open(CLEANED_CAPTIONS_PATH, 'rb') as f:
    cleaned_captions = pickle.load(f)

print(f"✅ Loaded captions for {len(cleaned_captions)} images.")

# ===== Step 2: Prepare text for tokenizer =====
all_captions = []
for caps in cleaned_captions.values():
    for c in caps:
        all_captions.append(f"<start> {c} <end>")

print(f"📝 Total captions: {len(all_captions)}")

# ===== Step 3: Tokenize =====
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)

# Save tokenizer
with open(TOKENIZER_PATH, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"[SAVED] Tokenizer -> {TOKENIZER_PATH}")

# ===== Step 4: Calculate max length =====
sequences = tokenizer.texts_to_sequences(all_captions)
MAX_LENGTH = max(len(seq) for seq in sequences)
print(f"📏 Max caption length: {MAX_LENGTH}")

# ===== (Optional) Pad sequences example =====
padded_example = pad_sequences(sequences[:3], maxlen=MAX_LENGTH, padding='post')
print("Example padded sequences (first 3):")
print(padded_example)
