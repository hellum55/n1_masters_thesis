import torchvision.transforms as T
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
from config import CFG


def get_autoaugment_transforms():
    """
    Returns torchvision AutoAugment-based train and validation transforms.
    Uses ImageNet policy by default.
    """

    train_transforms = T.Compose([
        T.Resize((CFG.HEIGHT, CFG.WIDTH)),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
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
