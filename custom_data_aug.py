import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from config import CFG

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
        T.RandomRotation(10),
        #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        T.RandomVerticalFlip(),
        #T.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, val_transforms
