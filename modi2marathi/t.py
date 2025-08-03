

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ModiOCRDataset, PadCollate, ResizeKeepAspect
from model import OCRModel

# ========== ✅ Detect Colab ==========
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ========== ✅ Set Base Directory ==========
if IN_COLAB:
    from google.colab import drive
    # drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/modi2marathi"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== ✅ Add to sys.path ==========
sys.path.append(BASE_DIR)

# ========== ✅ Define Paths ==========
IMAGE_DIR = os.path.join(BASE_DIR, "modi_dataset/images")
LABEL_DIR = os.path.join(BASE_DIR, "modi_dataset/labels")
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "modi_dataset/char_to_idx.json")
IDX_TO_CHAR_PATH = os.path.join(BASE_DIR, "modi_dataset/idx_to_char.json")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ========== ✅ Checkpoint Strategy ==========
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "ocr_model_latest.pt")

def get_checkpoint_path(epoch):
    return os.path.join(CHECKPOINT_DIR, f"ocr_model_epoch_{epoch}.pt")

# ========== ✅ Training Config ==========
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count()

# ========== ✅ Load Vocabulary ==========
with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)
print("====================>",len(char_to_idx),"<===============" )
num_classes = len(char_to_idx)  # +1 for CTC blank

# ========== ✅ Dataset ==========
transform = transforms.Compose([
    ResizeKeepAspect(target_height=64, max_width=256)
])

dataset = ModiOCRDataset(
    image_dir=IMAGE_DIR,
    label_dir=LABEL_DIR,
    transform=transform
)

max_len = max(len(sample["label"]) for sample in dataset)

print("📏 Max label length=======>:", max_len)


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=PadCollate(),
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ========== ✅ Model Setup ==========
model = OCRModel(img_height=128, num_classes=num_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

# ========== ✅ Resume from Checkpoint ==========
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
        outputs = model(images)

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

    # ====== Save Checkpoints ======
    epoch_ckpt_path = get_checkpoint_path(epoch)
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint_data, epoch_ckpt_path)
    torch.save(checkpoint_data, LATEST_CHECKPOINT)

    print(f"💾 Saved checkpoint: {epoch_ckpt_path}")
