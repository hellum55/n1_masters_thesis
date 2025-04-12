import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["file_name"]).convert("RGB")
        label = torch.tensor(row["label_idx"]).long()

        if self.transform:
            image = self.transform(image)

        return image, label


def load_dataset(csv_path: str, image_dir: str) -> pd.DataFrame:
    """
    Loads the filtered dataframe, appends full image paths, encodes labels,
    and checks for missing files.

    Returns:
        df: Pandas DataFrame with encoded labels and image paths
        idx_to_label: Dictionary mapping encoded label indices back to strings
    """
    df = pd.read_csv(csv_path)

    # Add image extension and full paths
    df["file_name"] = df["file_name"].astype(str) + ".png"
    df["file_name"] = df["file_name"].apply(lambda x: os.path.join(image_dir, x))

    # Optional: Check for missing image files
    missing = df[~df["file_name"].apply(os.path.exists)]
    if not missing.empty:
        print(f"[Warning] Missing images: {len(missing)}")

    # Label encode
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["label"])

    idx_to_label = dict(enumerate(le.classes_))
    return df, idx_to_label
