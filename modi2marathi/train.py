import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ModiOCRDataset, PadCollate, ResizeKeepAspect
from model import OCRModel
from config import (
    BASE_DIR, IMAGE_DIR, LABEL_DIR,
    CHAR_TO_IDX, IDX_TO_CHAR, CHAR_TO_IDX_PATH,
    LATEST_CHECKPOINT, get_checkpoint_path,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_HEIGHT,
    IMG_WIDTH, DEVICE, NUM_WORKERS, NUM_CLASSES, CHECKPOINT_DIR
)

# ========== ✅ Device ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

# ========== ✅ Load Vocabulary ==========
with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)

pad_idx = char_to_idx["<pad>"]
sos_idx = char_to_idx["<sos>"]
eos_idx = char_to_idx["<eos>"]
num_classes = len(char_to_idx)

# ========== ✅ Data ==========
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
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=PadCollate(pad_token=pad_idx, sos_token=sos_idx, eos_token=eos_idx),
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ========== ✅ Model, Loss, Optimizer ==========
model = OCRModel(vocab_size=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========== ✅ Resume Checkpoint ==========
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
start_epoch = 1
if os.path.exists(LATEST_CHECKPOINT):
    print(f"🔄 Resuming from checkpoint: {LATEST_CHECKPOINT}")
    checkpoint = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
else:
    print("🆕 Starting fresh training...")

# ========== ✅ Training ==========
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(DEVICE)          # [B, 1, H, W]
        labels = batch["label"].to(DEVICE)          # [B, T]

        # Teacher Forcing:
        tgt_input = labels[:, :-1]                  # Input sequence: <sos> + ...
        tgt_output = labels[:, 1:]                  # Target: ... + <eos>

        optimizer.zero_grad()
        logits = model(images, tgt_input)           # [B, T, vocab_size]

        # Flatten for loss: (B * T, vocab_size), (B * T)
        loss = criterion(logits.reshape(-1, num_classes), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"📅 Epoch [{epoch}/{EPOCHS}], 🔻 Loss: {avg_loss:.4f}")

    # Save Checkpoints
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint_data, get_checkpoint_path(epoch))
    torch.save(checkpoint_data, LATEST_CHECKPOINT)
    print(f"💾 Saved checkpoint: epoch {epoch}")

