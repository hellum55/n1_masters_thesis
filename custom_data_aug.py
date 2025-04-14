# Custom Canny Edge Transform
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from config import CFG
import random
import cv2
import numpy as np
from PIL import Image

class CannyEdgeTransform:
    def __init__(self, p=0.3, threshold1=100, threshold2=200):
        self.p = p
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def __call__(self, img):
        if random.random() < self.p:
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, self.threshold1, self.threshold2)
            edges_rgb = np.stack([edges]*3, axis=-1)
            img = Image.fromarray(edges_rgb)
        return img

def get_transforms(height=CFG.HEIGHT, width=CFG.WIDTH):
    """
    Returns train and validation torchvision transforms.

    Args:
        height: target image height
        width: target image width

    Returns:
        train_transforms, val_transforms (torchvision.transforms.Compose)
    """

    train_transforms = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(),
        #T.RandomRotation(10),
        #T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=15),
        CannyEdgeTransform(p=0.3),  # ✅ Inject Canny edge augmentation
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, val_transforms
