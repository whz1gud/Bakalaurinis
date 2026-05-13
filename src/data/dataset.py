"""
DISC21 Dataset classes for training and evaluation.

Training: Uses images with augmentations to create positive pairs
Evaluation: Loads queries and references separately for retrieval evaluation
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class DISC21TrainDataset(Dataset):
    """
    Training dataset for metric learning.
    
    Each sample returns (image, label) where label is the image class/identity.
    The metric learning loss functions handle pair/triplet mining automatically.
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            data_dir: Path to training images folder (e.g., data/train/)
            transform: Torchvision transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.transform = transform or self._default_transform()
        
        # Find all images
        self.image_paths = []
        self.labels = []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        
        for idx, img_file in enumerate(sorted(self.data_dir.glob('*'))):
            if img_file.suffix.lower() in valid_extensions:
                self.image_paths.append(img_file)
                # Each image is its own class for metric learning
                # (positive pairs created via augmentations)
                self.labels.append(idx)
        
        print(f"[DISC21TrainDataset] Found {len(self.image_paths)} images in {data_dir}")
    
    def _default_transform(self) -> transforms.Compose:
        """Default training transforms with augmentations."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]


class DISC21PairDataset(Dataset):
    """
    Dataset that creates positive pairs using augmentations.
    
    For each image, applies two different augmentations to create a positive pair.
    Useful for contrastive learning approaches.
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform or self._default_transform()
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        self.image_paths = []
        
        for img_file in sorted(self.data_dir.glob('*')):
            if img_file.suffix.lower() in valid_extensions:
                self.image_paths.append(img_file)
        
        print(f"[DISC21PairDataset] Found {len(self.image_paths)} images in {data_dir}")
    
    def _default_transform(self) -> transforms.Compose:
        """Strong augmentations for creating positive pairs."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Returns (augmented_view_1, augmented_view_2, label)"""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform twice to get two different augmented views
        view1 = self.transform(image)
        view2 = self.transform(image)
        
        return view1, view2, idx


class DISC21EvalDataset(Dataset):
    """
    Evaluation dataset for image retrieval.
    
    Loads either query images or reference images for evaluation.
    Supports nested folder structures (uses recursive glob).
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            data_dir: Path to images folder (queries or refs)
            transform: Torchvision transforms (should be deterministic for eval)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform or self._default_transform()
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        self.image_paths = []
        self.image_ids = []
        
        # Use recursive glob to find images in nested folders too
        for img_file in sorted(self.data_dir.rglob('*')):
            if img_file.suffix.lower() in valid_extensions:
                self.image_paths.append(img_file)
                # Extract ID from filename (e.g., "Q00003.jpg" -> "Q00003")
                self.image_ids.append(img_file.stem)
        
        print(f"[DISC21EvalDataset] Found {len(self.image_paths)} images in {data_dir}")
    
    def _default_transform(self) -> transforms.Compose:
        """Deterministic transforms for evaluation."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Returns (image_tensor, image_id)"""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.image_ids[idx]


def load_groundtruth(csv_path: str) -> Dict[str, str]:
    """
    Load ground truth mappings from CSV.
    
    Args:
        csv_path: Path to groundtruth CSV file
        
    Returns:
        Dictionary mapping query_id -> reference_id
    """
    groundtruth = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                query_id, ref_id = row[0], row[1]
                groundtruth[query_id] = ref_id
    
    print(f"[load_groundtruth] Loaded {len(groundtruth)} query-reference pairs from {csv_path}")
    return groundtruth


def get_eval_transforms() -> transforms.Compose:
    """Get standard evaluation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_train_transforms() -> transforms.Compose:
    """Get training transforms with augmentations."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


