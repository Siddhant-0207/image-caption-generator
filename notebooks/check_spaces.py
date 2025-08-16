path = "data/Flickr8k_text/caption_file/captions.txt"
with open(path, "r", encoding="utf-8") as f:
    first_line = f.readline()
print(repr(first_line))
