import os
from PIL import Image

img_size = (64, 64)

data_dir = "./dataset/chibi"

if not os.path.exists("./dataset/chibi_resized"):
    os.makedirs("./dataset/chibi_resized")

for filename in os.listdir(data_dir):
    img = Image.open(os.path.join(data_dir, filename))

    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize the image
    img = img.resize(img_size)

    # Save the resized image to the new directory
    img.save(os.path.join("./dataset/chibi_resized", filename))
