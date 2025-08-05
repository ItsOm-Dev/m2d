import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# ========== ✅ Local Imports ==========
from dataset import ModiOCRDataset, PadCollate, ResizeKeepAspect
from model import OCRModel
from config import (
    BASE_DIR, IMAGE_DIR, LABEL_DIR,
    CHAR_TO_IDX, IDX_TO_CHAR, CHAR_TO_IDX_PATH,
    LATEST_CHECKPOINT, get_checkpoint_path,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_HEIGHT,
    IMG_WIDTH, DEVICE, NUM_WORKERS, NUM_CLASSES,CHECKPOINT_DIR
)
# ========== ✅ Device Setup ==========
# if(torch.cuda.is_available()):
#     print("Cuudddaaaaa........\n")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = os.cpu_count()

# ========== ✅ Checkpoint Setup ==========
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "ocr_model_latest.pt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_checkpoint_path(epoch):
    return os.path.join(CHECKPOINT_DIR, f"ocr_model_epoch_{epoch}.pt")

# ========== ✅ Load Vocabulary ==========
with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)
num_classes = len(char_to_idx)  # No need to add 1, handled in model
print(f"✅ Character classes: {num_classes}")

# ========== ✅ Data Pipeline ==========
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
    collate_fn=PadCollate(),
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ========== ✅ Model Setup ==========
model = OCRModel(num_classes=num_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

# ========== ✅ Resume Checkpoint ==========
start_epoch = 1
if os.path.exists(LATEST_CHECKPOINT):
    print(f"🔄 Resuming from checkpoint: {LATEST_CHECKPOINT}")
    checkpoint = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
else:
    print("🆕 Starting fresh training...")

# ========== ✅ Training Loop ==========
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        label_lens = batch["label_len"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)  # [T, B, C]
        # === 🔍 Debug time steps vs label length ===
        # if epoch == start_epoch:
        #     print("👀 Model output shape (T, B, C):", outputs.shape)
        #     print("📏 Max label length in batch:", label_lens.max().item())

        input_lens = torch.full(
            size=(outputs.size(1),),
            fill_value=outputs.size(0),
            dtype=torch.long
        ).to(DEVICE)

        loss = criterion(outputs.log_softmax(2), labels, input_lens, label_lens)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"📅 Epoch [{epoch}/{EPOCHS}], 🔻 Loss: {avg_loss:.4f}")

    # ====== ✅ Save Checkpoints ======
    epoch_ckpt_path = get_checkpoint_path(epoch)
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint_data, epoch_ckpt_path)
    torch.save(checkpoint_data, LATEST_CHECKPOINT)

    print(f"💾 Saved checkpoint: {epoch_ckpt_path}")
