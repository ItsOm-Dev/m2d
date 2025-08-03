import os
import json
from collections import Counter

LABEL_DIR = "modi_dataset_syn/labels"
all_chars = Counter()

for fname in os.listdir(LABEL_DIR):
    with open(os.path.join(LABEL_DIR, fname), "r", encoding="utf-8") as f:
        text = f.read().strip()
        all_chars.update(text)

# Build char-to-index mapping
chars = sorted(all_chars.keys())
char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}  # 0 is reserved for CTC blank
idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}

with open("modi_dataset_syn/char_to_idx.json", "w", encoding="utf-8") as f:
    json.dump(char_to_idx, f, ensure_ascii=False, indent=2)
with open("modi_dataset_syn/idx_to_char.json", "w", encoding="utf-8") as f:
    json.dump(idx_to_char, f, ensure_ascii=False, indent=2)

print("✅ char_to_idx and idx_to_char saved.")
