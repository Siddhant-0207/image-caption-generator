import os
import pickle
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np

# Paths
IMAGES_PATH = "data/Flickr8k_Dataset/Images"
OUTPUT_PATH = os.path.join("data", "processed", "image_features.pkl")

# Create output folder if not exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load InceptionV3 model (without final classification layer)
base_model = InceptionV3(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def extract_features(filename, model):
    img = load_img(filename, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()

# Extract features for all images
features = {}
for img_name in tqdm(os.listdir(IMAGES_PATH)):
    img_path = os.path.join(IMAGES_PATH, img_name)
    features[img_name] = extract_features(img_path, model)

# Save features
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(features, f)

print(f"[SAVED] Image features to {OUTPUT_PATH}")
