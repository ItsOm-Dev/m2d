import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torchvision.transforms import functional as TF
import torch.nn.functional as F

# ========== ✅ Config Imports ==========
from config import (
    IMAGE_DIR, LABEL_DIR,
    CHAR_TO_IDX_PATH,
    MAX_LABEL_LENGTH
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

        # ✅ Special token indices
        self.pad_idx = self.char_to_idx["<pad>"]
        self.sos_idx = self.char_to_idx["<sos>"]
        self.eos_idx = self.char_to_idx["<eos>"]

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
    
        # ✅ Encode label with <sos> and <eos>
        label = [self.sos_idx] + [self.char_to_idx.get(c, self.pad_idx) for c in text] + [self.eos_idx]
        label = label[:MAX_LABEL_LENGTH]  # Truncate if needed

    # return image, torch.tensor(label, dtype=torch.long)


        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "label_len": len(label),
            "image_name": image_file
        }

# ========== ✅ Collate Function ==========
class PadCollate:
    def __init__(self, pad_token=None, sos_token=None, eos_token=None):
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token


    def __call__(self, batch):
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]
        image_names = [item["image_name"] for item in batch]

        # Stack image tensors
        images = torch.stack(images)

        # Pad label sequences
        max_len = max(len(label) for label in labels)
        padded_labels = []
        for label in labels:
            padded = F.pad(label, (0, max_len - len(label)), value=self.pad_token)
            padded_labels.append(padded)
        labels = torch.stack(padded_labels)

        return {
            "image": images,        # [B, 1, H, W]
            "label": labels,        # [B, T]
            "image_name": image_names
        }


# ========== ✅ Resize + Pad Image (Keep Aspect) ==========
class ResizeKeepAspect:
    def __init__(self, target_height=128, max_width=1268):
        self.target_height = target_height
        self.max_width = max_width

    def __call__(self, img):
        img = img.convert("L")  # Ensure grayscale
        orig_w, orig_h = img.size
        new_w = int(orig_w * self.target_height / orig_h)
        img = TF.resize(img, (self.target_height, new_w))

        if new_w < self.max_width:
            pad = self.max_width - new_w
            img = TF.pad(img, (0, 0, pad, 0), fill=255)
        else:
            img = TF.crop(img, 0, 0, self.target_height, self.max_width)

        return TF.to_tensor(img)

