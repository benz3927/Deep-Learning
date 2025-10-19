import os
import random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import pytorch_lightning as pl

class OxfordPetDataset(Dataset):
    def __init__(self, root, split='trainval', img_size=512, classes='trimap', transform=None):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.classes = classes
        self.transform = transform

        # Load filenames from split file
        split_file = self.root / 'annotations' / f'{split}.txt'
        with open(split_file) as f:
            # IMPORTANT: Split files contain "filename class_id species breed_id"
            # We only need the first column (filename)
            lines = f.readlines()
            self.filenames = []
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    # Take only the first part before any space
                    filename = line.split()[0]
                    self.filenames.append(filename)

        self.img_dir = self.root / 'images'
        self.mask_dir = self.root / 'annotations' / 'trimaps'

    def __len__(self):
        return len(self.filenames)

    def mask_to_classes(self, mask: Image.Image):
        """Convert trimap values (1,2,3) to class indices (0,1,2)"""
        m = np.array(mask, dtype=np.int64)
        if self.classes == 'trimap':
            m = m - 1  # 0=pet, 1=background, 2=border
        else:
            # Binary: pet vs background (merge border with pet)
            pet = (m == 1) | (m == 3)
            m = pet.astype(np.int64)
        return torch.tensor(m, dtype=torch.long)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = self.img_dir / f"{name}.jpg"
        mask_path = self.mask_dir / f"{name}.png"

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Resize image and mask
        # CRITICAL: Use NEAREST interpolation for masks to preserve class labels
        img = TF.resize(img, [self.img_size, self.img_size], 
                       interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], 
                        interpolation=Image.NEAREST)

        # Convert mask to class indices
        mask = self.mask_to_classes(mask)

        # Apply transforms to image (ToTensor + Normalize)
        if self.transform:
            img = self.transform(img)

        return img, mask

class OxfordPetDataModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=8, img_size=512, num_workers=4, classes='trimap'):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.classes = classes

    def setup(self, stage=None):
        # Image transforms (normalization with ImageNet stats)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Load full trainval dataset
        full_ds = OxfordPetDataset(
            self.root, 
            split='trainval',
            img_size=self.img_size,
            classes=self.classes,
            transform=transform
        )
        
        # Split train/val 80/20 with fixed seed for reproducibility
        n_train = int(0.8 * len(full_ds))
        n_val = len(full_ds) - n_train
        self.train_ds, self.val_ds = random_split(
            full_ds, 
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        # Test dataset
        self.test_ds = OxfordPetDataset(
            self.root, 
            split='test',
            img_size=self.img_size,
            classes=self.classes,
            transform=transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )