import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ==== PATHS ====
MODEL_PATH = os.path.join("src", "model", "caption_model.keras")  # trained model
TOKENIZER_PATH = "data/processed/tokenizer.pkl"                   # tokenizer
CAPTIONS_PATH = "data/processed/captions_cleaned.pkl"            # cleaned captions
MAX_LENGTH = 34                                                   # max caption length
FEATURE_SIZE = 2048                                               # InceptionV3 feature size

# ==== LOAD MODEL & TOKENIZER ====
print("📂 Loading trained model...")
model = load_model(MODEL_PATH)

print("📂 Loading tokenizer...")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("📂 Loading captions...")
with open(CAPTIONS_PATH, "rb") as f:
    cleaned_captions = pickle.load(f)

# ==== LOAD FEATURE EXTRACTOR ====
print("📂 Loading InceptionV3 feature extractor...")
cnn_model = InceptionV3(weights="imagenet")
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)  # last pooling layer

def extract_features(image_path):
    """Extract CNN features from an image."""
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = feature_extractor.predict(img, verbose=0)
    return features

def word_for_id(integer, tokenizer):
    """Map an integer to a word from tokenizer."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(photo_features):
    """Generate caption from extracted features."""
    in_text = "<start>"
    for _ in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
        
        yhat = model.predict([photo_features, sequence], verbose=0)
        yhat = np.argmax(yhat)  # pick highest probability word
        
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        if word == "<end>" or word == "end":  # check both possibilities
            break
        in_text += " " + word
        if word == "<end>":
            break
    
    # clean the output
    final_caption = in_text.replace("<start>", "").replace("<end>", "").strip()
    return final_caption

# ==== TEST WITH SAMPLE IMAGE ====
if __name__ == "__main__":
    test_image_path = r"C:/sidd/Image caption generator/data/Flickr8k_Dataset/Images/95734038_2ab5783da7.jpg"
    image_id = os.path.basename(test_image_path)  # filename for captions lookup

    print(f"🖼 Extracting features for {test_image_path}...")
    photo_features = extract_features(test_image_path)
    
    print("📝 Generating caption...")
    predicted_caption = generate_caption(photo_features)
    
    # Show original caption if available
    if image_id in cleaned_captions:
        original_caption = cleaned_captions[image_id][0]  # first caption
        print("\n💬 Original Caption:", original_caption)
    
    print("\n🎯 Predicted Caption:", predicted_caption)
