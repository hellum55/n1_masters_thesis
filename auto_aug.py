import torchvision.transforms as T
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
from config import CFG

from torchvision.transforms import InterpolationMode
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

def get_autoaugment_transforms():
    """
    Returns torchvision AutoAugment-based train and validation transforms.
    Uses ImageNet policy by default.
    """

    train_transforms = T.Compose([
        T.Resize((CFG.HEIGHT, CFG.WIDTH)),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        CannyEdgeTransform(p=0.4),  # ✅ Inject Canny edge augmentation
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = T.Compose([
        T.Resize((CFG.HEIGHT, CFG.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, val_transforms
