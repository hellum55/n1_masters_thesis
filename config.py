import os
import torch

class CFG:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_DEVICES = torch.cuda.device_count()
    NUM_WORKERS = os.cpu_count()
    NUM_CLASSES = 34
    EPOCHS = 3
    BATCH_SIZE = (
        64 if torch.cuda.device_count() < 2 
        else (64 * torch.cuda.device_count())
    )
    TEST_SIZE = 0.1
    LR = 0.0005
    APPLY_SHUFFLE = True
    SEED = 768
    HEIGHT = 224
    WIDTH = 224
    CHANNELS = 3
    IMAGE_SIZE = (224, 224, 3)