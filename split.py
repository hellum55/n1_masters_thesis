from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import CFG
from load_data import ImageDataset

def get_dataloaders(df, train_transforms, val_transforms):
    """
    Returns both datasets and dataloaders.
    """

    train_df, val_df = train_test_split(
        df,
        test_size=CFG.TEST_SIZE,
        stratify=df["label_idx"],
        random_state=CFG.SEED
    )

    train_dataset = ImageDataset(train_df, transform=train_transforms)
    val_dataset = ImageDataset(val_df, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=CFG.APPLY_SHUFFLE,
        num_workers=CFG.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS
    )

    return train_dataset, val_dataset, train_loader, val_loader
