import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ModiOCRDataset, PadCollate, ResizeKeepAspect
from model import OCRModel
from config import (
    IMAGE_DIR, LABEL_DIR, CHAR_TO_IDX_PATH, IDX_TO_CHAR,
    LATEST_CHECKPOINT, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH,
    DEVICE, NUM_WORKERS
)

# ========== ✅ Load Vocab ==========
with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)

idx_to_char = {int(k): v for k, v in json.load(open(IDX_TO_CHAR, encoding="utf-8")).items()}
pad_idx = char_to_idx["<pad>"]
sos_idx = char_to_idx["<sos>"]
eos_idx = char_to_idx["<eos>"]
vocab_size = len(char_to_idx)

# ========== ✅ Dataset ==========
transform = transforms.Compose([
    ResizeKeepAspect(target_height=IMG_HEIGHT, max_width=IMG_WIDTH)
])

dataset = ModiOCRDataset(
    image_dir=IMAGE_DIR,
    label_dir=LABEL_DIR,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=PadCollate(pad_token=pad_idx, sos_token=sos_idx, eos_token=eos_idx),
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ========== ✅ Model ==========
model = OCRModel(vocab_size=vocab_size).to(DEVICE)
model.eval()

if os.path.exists(LATEST_CHECKPOINT):
    checkpoint = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Loaded checkpoint from: {LATEST_CHECKPOINT}")
else:
    raise FileNotFoundError(f"No checkpoint found at: {LATEST_CHECKPOINT}")

# ========== ✅ Inference Loop ==========
def decode_prediction(indices):
    tokens = []
    for idx in indices:
        char = idx_to_char.get(idx, "")
        if char == "<eos>":
            break
        if char not in ("<sos>", "<pad>"):
            tokens.append(char)
    return "".join(tokens)

MAX_LEN = 100

print("\n🧪 Evaluating on test set:\n")
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        image = batch["image"].to(DEVICE)       # [1, 1, H, W]
        true_label = batch["label"][0].tolist() # [T]

        # Start token
        generated = [sos_idx]

        for _ in range(MAX_LEN):
            tgt_input = torch.tensor(generated, dtype=torch.long, device=DEVICE).unsqueeze(0)  # [1, T]
            logits = model(image, tgt_input)  # [1, T, vocab_size]
            next_token = logits[0, -1].argmax(-1).item()  # get last token

            if next_token == eos_idx:
                break
            generated.append(next_token)

        pred_text = decode_prediction(generated)
        true_text = decode_prediction(true_label)

        print(f"🔡 Prediction: {pred_text}")
        print(f"🎯 Ground Truth: {true_text}")
        print("-" * 60)

        if i == 9:
            break  # limit output

