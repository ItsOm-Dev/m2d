import os
import json
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import ModiOCRDataset, PadCollate, ResizeKeepAspect
from config import IMAGE_DIR, LABEL_DIR, CHAR_TO_IDX_PATH, IMG_HEIGHT, IMG_WIDTH

# Load char_to_idx just for dataset init
with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)

# Transform: resize + pad
transform = transforms.Compose([
    ResizeKeepAspect(target_height=IMG_HEIGHT, max_width=IMG_WIDTH)
])

# Create dataset and loader
dataset = ModiOCRDataset(
    image_dir=IMAGE_DIR,
    label_dir=LABEL_DIR,
    transform=transform
)
loader = DataLoader(dataset, batch_size=8, collate_fn=PadCollate())

# Visualize first batch
for batch in loader:
    images = batch["image"]  # [B, 1, H, W]
    image_names = batch["image_name"]

    # Convert to numpy and plot
    for i in range(len(images)):

    # for i in range(min(4, len(images))):
        img_tensor = images[i].squeeze(0)  # [H, W]
        plt.figure(figsize=(10, 2))
        plt.imshow(img_tensor, cmap='gray')
        plt.title(f"{image_names[i]} — shape: {img_tensor.shape}")
        plt.axis('off')
        plt.show()
    break
