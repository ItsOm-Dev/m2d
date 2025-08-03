import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# ========== ✅ Local Imports ==========
from dataset import ModiOCRDataset, PadCollate, ResizeKeepAspect
from model import OCRModel
from config import (
    BASE_DIR, IMAGE_DIR, LABEL_DIR,
    CHAR_TO_IDX, IDX_TO_CHAR_PATH, CHAR_TO_IDX_PATH,
    LATEST_CHECKPOINT, get_checkpoint_path,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_HEIGHT,
    IMG_WIDTH, DEVICE, NUM_WORKERS, NUM_CLASSES,CHECKPOINT_DIR
)
# ========== ✅ Checkpoint Path ==========
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "ocr_model_latest.pt")

# ========== ✅ Device ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== ✅ Load Vocabulary ==========
with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)
with open(IDX_TO_CHAR_PATH, "r", encoding="utf-8") as f:
    idx_to_char = {int(k): v for k, v in json.load(f).items()}

num_classes = len(char_to_idx)
print(f"🔠 Loaded vocab with {num_classes} characters")

# ========== ✅ Transform ==========
transform = transforms.Compose([
    ResizeKeepAspect(target_height=IMG_HEIGHT, max_width=IMG_WIDTH)
])

# ========== ✅ Dataset ==========
dataset = ModiOCRDataset(
    image_dir=IMAGE_DIR,
    label_dir=LABEL_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=PadCollate()
)

# ========== ✅ Load Model ==========
model = OCRModel(num_classes=num_classes).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ========== ✅ Decoder ==========
def ctc_greedy_decoder(output, idx_to_char):
    output = output.permute(1, 0, 2)  # [B, T, C]
    pred_texts = []
    for sequence in output:
        best_path = torch.argmax(sequence, dim=1).tolist()
        decoded = []
        prev = None
        for idx in best_path:
            if idx != 0 and idx != prev:
                decoded.append(idx_to_char.get(idx, ""))
            prev = idx
        pred_texts.append("".join(decoded))
    return pred_texts

# ========== ✅ Evaluation ==========
print("📊 Evaluation Results:\n")

with torch.no_grad():
    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"]
        image_names = batch["image_name"]

        output = model(images)
        # === 🔍 Debug time steps vs label length ===
        print("👀 Eval Output shape (T, B, C):", output.shape)
        print("📏 Max GT label length in batch:", max(len(l) for l in labels))

        decoded_preds = ctc_greedy_decoder(output, idx_to_char)

        for i, pred in enumerate(decoded_preds):
            gt_indices = labels[i].tolist()
            gt = "".join([idx_to_char.get(idx, "") for idx in gt_indices if idx != 0])
            print(f"🖼️ {image_names[i]}:\n   🔹 GT:  {gt}\n   🔸 Pred: {pred}\n")
