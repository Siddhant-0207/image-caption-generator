path = "data/Flickr8k_text/caption_file/captions.txt"

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print("First 5 lines:")
for l in lines[:5]:
    print(repr(l))
