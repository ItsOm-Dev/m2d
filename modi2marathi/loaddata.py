from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm

SAVE_DIR = "modi_dataset_syn"
IMAGE_DIR = os.path.join(SAVE_DIR, "images")
LABEL_DIR = os.path.join(SAVE_DIR, "labels")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

dataset = load_dataset("historyHulk/SynthMoDe", split="train")
print("Dataset keys:", dataset[0].keys())  # Should show image1, image2, text

for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
    base_id = f"image_{idx+1:05d}"
    label_text = sample["text"].strip()

    # Save image1 (font 1)
    img1_path = os.path.join(IMAGE_DIR, f"{base_id}.png")
    lbl1_path = os.path.join(LABEL_DIR, f"{base_id}.txt")
    sample["image1"].save(img1_path)
    with open(lbl1_path, "w", encoding="utf-8") as f:
        f.write(label_text)

    # Save image2 (font 2)
    img2_path = os.path.join(IMAGE_DIR, f"{base_id}_font2.png")
    lbl2_path = os.path.join(LABEL_DIR, f"{base_id}_font2.txt")
    sample["image2"].save(img2_path)
    with open(lbl2_path, "w", encoding="utf-8") as f:
        f.write(label_text)

print("✅ Dataset successfully saved in modi_dataset_syn format!")
