import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ==== PATHS ====
MODEL_PATH = os.path.join("src", "model", "caption_model.keras")   # Path to your trained model
TOKENIZER_PATH = os.path.join("data", "processed", "tokenizer.pkl") # Tokenizer file
MAX_LENGTH = 34                                                    # Same as during training
FEATURE_SIZE = 2048                                                # InceptionV3 output size

# ==== 1. LOAD MODEL & TOKENIZER ====
print("📂 Loading trained model...")
model = load_model(MODEL_PATH)

print("📂 Loading tokenizer...")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# ==== 2. LOAD FEATURE EXTRACTOR ====
print("📂 Loading InceptionV3 feature extractor...")
cnn_model = InceptionV3(weights="imagenet")
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)  # Last pooling layer

def extract_features(image_path):
    """Extract CNN features from an image."""
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = feature_extractor.predict(img, verbose=0)
    return features

# ==== 3. ID TO WORD ====
def word_for_id(integer, tokenizer):
    """Map an integer to a word from tokenizer."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# ==== 4. CAPTION GENERATOR ====
def generate_caption(photo_features):
    """Generate caption from extracted features."""
    in_text = "<start>"
    for _ in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
        
        yhat = model.predict([photo_features, sequence], verbose=0)
        yhat = np.argmax(yhat)  # Pick highest probability word
        
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        if word == "<end>" or word == "end":  # check both possibilities
            break
        in_text += " " + word
    
    # Remove <start> and <end> tokens
    final_caption = in_text.replace("<start>", "").replace("<end>", "").strip()
    return final_caption

# ==== 5. TEST WITH SAMPLE IMAGE ====
if __name__ == "__main__":
    test_image_path = r"C:/sidd/Image caption generator/data/Flickr8k_Dataset/Images/95734035_84732a92c1.jpg"
    
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"❌ Image not found: {test_image_path}")
    
    print(f"🖼 Extracting features for {test_image_path}...")
    photo_features = extract_features(test_image_path)
    
    print("📝 Generating caption...")
    caption = generate_caption(photo_features)
    
    print("\n🎯 Final Caption:", caption)
