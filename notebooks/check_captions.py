import pickle

with open("data/processed/captions_cleaned.pkl", "rb") as f:
    captions = pickle.load(f)

print(len(captions))          # Should be ~8000 for Flickr8k
print(list(captions.items())[:2])  # Show a couple of items
