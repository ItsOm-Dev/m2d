import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torchvision.transforms import functional as TF

import sys

# ========== ✅ Environment Setup ==========
from config import (
    BASE_DIR, IMAGE_DIR, LABEL_DIR,
    CHAR_TO_IDX, IDX_TO_CHAR, CHAR_TO_IDX_PATH,
    LATEST_CHECKPOINT, get_checkpoint_path,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_HEIGHT,
    IMG_WIDTH, DEVICE, NUM_WORKERS, NUM_CLASSES,MAX_LABEL_LENGTH
)
# ========== ✅ Dataset Class ==========
class ModiOCRDataset(Dataset):
    def __init__(self, image_dir=IMAGE_DIR, label_dir=LABEL_DIR, transform=None, vocab_path=CHAR_TO_IDX_PATH):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.char_to_idx = json.load(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, image_file)

        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_file)

        image = Image.open(img_path).convert("L")
        image = self.transform(image) if self.transform else transforms.ToTensor()(image)

        with open(label_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        label = [self.char_to_idx.get(c, 0) for c in text][:MAX_LABEL_LENGTH]  # Truncate if needed

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "label_len": len(label),
            "image_name": image_file
        }

# ========== ✅ Collate Function ==========
class PadCollate:
    def __call__(self, batch):
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]
        label_lens = [item["label_len"] for item in batch]
        image_names = [item["image_name"] for item in batch]

        max_label_len = min(MAX_LABEL_LENGTH, max(label_lens))  # cap to config
        padded_labels = torch.zeros(len(labels), max_label_len, dtype=torch.long)
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = label[:max_label_len]

        return {
            "image": torch.stack(images),
            "label": padded_labels,
            "label_len": torch.tensor(label_lens, dtype=torch.long),
            "image_name": image_names
        }

# ========== ✅ Resize + Pad ==========
class ResizeKeepAspect:
    def __init__(self, target_height=128, max_width=1268):
        self.target_height = target_height
        self.max_width = max_width

    def __call__(self, img):
        img = img.convert("L") #Enures grayscale image
        orig_w, orig_h = img.size
        new_w = int(orig_w * self.target_height / orig_h)
        img = TF.resize(img, (self.target_height, new_w))

        if new_w < self.max_width:
            pad = self.max_width - new_w
            img = TF.pad(img, (0, 0, pad, 0), fill=255)
        else:
            img = TF.crop(img, 0, 0, self.target_height, self.max_width)

        return TF.to_tensor(img)
