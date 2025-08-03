import os
import json
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
# ========== ✅ Detect Colab ==========
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ========== ✅ Mount Google Drive if in Colab ==========
if IN_COLAB:
    from google.colab import drive
    # drive.mount('/content/drive', force_remount=True)
    BASE_DIR = "/content/drive/MyDrive/modi2marathi"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== ✅ Define Dataset and Checkpoint Paths ==========
##################################################################################################
#########  for Synthetic images
#######################CURRENT DATASET: SYNTHETIC (SynthMoDe)####################################
# IMAGE_DIR = os.path.join(BASE_DIR, "modi_dataset_syn/images")
# LABEL_DIR = os.path.join(BASE_DIR, "modi_dataset_syn/labels")
# CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "modi_dataset_syn/char_to_idx.json")
# IDX_TO_CHAR_PATH = os.path.join(BASE_DIR, "modi_dataset_syn/idx_to_char.json")


###################   Mixed Dataset  ################################

# IMAGE_DIR = os.path.join(BASE_DIR, "mixed/images")
# LABEL_DIR = os.path.join(BASE_DIR, "mixed/labels")
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "mixed/char_to_idx.json")
IDX_TO_CHAR_PATH = os.path.join(BASE_DIR, "mixed/idx_to_char.json")



###################   ORG Dataset  ################################

IMAGE_DIR = os.path.join(BASE_DIR, "modi_dataset/images")
LABEL_DIR = os.path.join(BASE_DIR, "modi_dataset/labels")
# CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "modi_dataset/char_to_idx.json")
# IDX_TO_CHAR_PATH = os.path.join(BASE_DIR, "modi_dataset/idx_to_char.json")
###############################################################################3
CHECKPOINT_DIR = os.path.join(BASE_DIR, "saved_models")
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "ocr_model_latest.pt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_checkpoint_path(epoch):
    return os.path.join(CHECKPOINT_DIR, f"ocr_model_epoch_{epoch}.pt")

# ========== ✅ Training Hyperparameters ==========
BATCH_SIZE = 4
EPOCHS = 25 # 45 WILL BE SEMI FINAL
LEARNING_RATE = 1e-4 # sset to 1e-4 if not work or loss function suddern JUMP
IMG_HEIGHT =128 #128 is best but
IMG_WIDTH =720 #1268 is best but 950
MAX_LABEL_LENGTH=452
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = multiprocessing.cpu_count()

# ========== ✅ Load Vocabulary ==========
with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
    CHAR_TO_IDX = json.load(f)

with open(IDX_TO_CHAR_PATH, "r", encoding="utf-8") as f:
    IDX_TO_CHAR = {int(k): v for k, v in json.load(f).items()}

NUM_CLASSES = len(CHAR_TO_IDX)  # No +1; blank=0 already handled by CTC setup
