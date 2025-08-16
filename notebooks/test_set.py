import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Base data folder
data_folder = r"C:\sidd\Image caption generator\data"  # change if your 'data' is elsewhere

# Try to locate the Images folder
images_folder = None
for root, dirs, files in os.walk(data_folder):
    if 'Images' in dirs:
        images_folder = os.path.join(root, 'Images')
        break

if images_folder is None:
    raise FileNotFoundError("❌ Could not find 'Images' folder inside data/")

# List all image files (jpg/png)
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    raise FileNotFoundError(f"❌ No image files found in {images_folder}")

# Pick a random image
random_image = random.choice(image_files)
img_path = os.path.join(images_folder, random_image)

# Load and display
img = Image.open(img_path)
plt.imshow(img)
plt.axis('off')
plt.title(random_image)
plt.show()
