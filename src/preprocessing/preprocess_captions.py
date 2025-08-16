import os
import string
import pickle

# ===== Paths according to your folder structure =====
RAW_CAPTIONS_PATH = "data/Flickr8k_text/caption_file/captions.txt"
PROCESSED_DIR = "data/processed"

OUTPUT_CLEANED_PATH = "data/processed/captions_cleaned.pkl"
OUTPUT_VOCAB_PATH = "data/processed/vocab.txt"


# ===== Step 1: Load captions =====
import csv 
def load_captions(filename):
    """
    Load raw captions from the dataset file.
    Returns dict: {image_id: [captions]}
    """
    captions = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for img_id, caption in reader:
            img_id = img_id.split('#')[0]  # remove #index if present
            captions.setdefault(img_id, []).append(caption)
    return captions


# ===== Step 2: Clean captions =====
def clean_caption(caption):
    """
    Lowercase, remove punctuation, remove single char words & numbers
    """
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = " ".join([word for word in caption.split() if len(word) > 1 and word.isalpha()])
    return caption


def clean_captions(captions_dict):
    """
    Apply cleaning to all captions
    """
    cleaned = {}
    for img_id, caps in captions_dict.items():
        cleaned[img_id] = [clean_caption(c) for c in caps]
    return cleaned


# ===== Step 3: Build vocabulary =====
def build_vocab(cleaned_captions):
    """
    Build a sorted vocabulary set from cleaned captions
    """
    vocab = set()
    for caps in cleaned_captions.values():
        for c in caps:
            vocab.update(c.split())
    return sorted(vocab)


# ===== Step 4: Save outputs =====
def save_pickle(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # ensure folder exists
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"[SAVED] {filename}")


def save_vocab(vocab, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(f"{word}\n")
    print(f"[SAVED] {filename}")


# ===== Main =====
if __name__ == "__main__":
    print("📥 Loading captions...")
    captions = load_captions(RAW_CAPTIONS_PATH)
    print(f"✅ Loaded {len(captions)} images with captions.")

    print("🧹 Cleaning captions...")
    cleaned = clean_captions(captions)

    print("🛠 Building vocabulary...")
    vocab = build_vocab(cleaned)
    print(f"✅ Vocabulary size: {len(vocab)} words.")

    print("💾 Saving processed files...")
    save_pickle(cleaned, OUTPUT_CLEANED_PATH)
    save_vocab(vocab, OUTPUT_VOCAB_PATH)

    print("🎯 Preprocessing complete!")
