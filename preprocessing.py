import cv2
import numpy as np
import torch

def preprocess_image(image_path, img_height=128, img_width=512):
    # 1. Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. Denoise
    img = cv2.fastNlMeansDenoising(img, None, h=30)

    # 3. Binarize (adaptive threshold works better on uneven lighting)
    img = cv2.adaptiveThreshold(img, 255, 
                                cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY_INV, 
                                blockSize=15, C=11)

    # 4. Deskew (optional, based on largest contour or Hough lines)
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, 
                         borderMode=cv2.BORDER_REPLICATE)

    # 5. Resize with aspect ratio preserved
    h, w = img.shape
    new_w = int(img_height * w / h)
    img = cv2.resize(img, (new_w, img_height))

    # Pad to fixed size (centered)
    if new_w > img_width:
        img = cv2.resize(img, (img_width, img_height))
    else:
        padded_img = np.zeros((img_height, img_width), dtype=np.uint8)
        x_offset = (img_width - new_w) // 2
        padded_img[:, x_offset:x_offset + new_w] = img
        img = padded_img

    # 6. Normalize and add channel dimension
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # shape = [1, H, W]
    
    # 7. Convert to tensor
    return torch.tensor(img)
