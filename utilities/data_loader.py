import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class BinaryLabelTransform:
    """
    A custom transform that converts any pixel > 0.5 to 1.0, else 0.0.
    Ensures labels are strictly 0 or 1.
    """
    def __call__(self, x):
        return torch.where(x > 0.5, 1.0, 0.0)

class ChangeDetectionDataset(Dataset):
    """
    A dataset class for change detection with two time-phase images (A and B)
    and a corresponding binary label.
    """
    def __init__(self, root_dir, use_augmentation=False, image_size=(512,512)):
        """
        Args:
            root_dir (str): Directory with subfolders 'A', 'B', and 'label'.
            use_augmentation (bool): Whether to apply data augmentation.
            image_size (tuple): (height, width) to resize images.
        """
        self.root_dir = root_dir
        self.use_augmentation = use_augmentation
        self.image_size = image_size
        
        # Paths for images A, B, and labels
        self.image_A_dir = os.path.join(root_dir, 'A')
        self.image_B_dir = os.path.join(root_dir, 'B')
        self.label_dir = os.path.join(root_dir, 'label')
        self.image_list = [
            f for f in os.listdir(self.image_A_dir) 
            if f.endswith(('.jpg', '.png'))
        ]
        
        # Base transforms for resizing and converting to tensor
        self.base_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
        ])

        # Normalization
        self.normalize = T.Normalize(
            mean=[123.675/255.0, 116.28/255.0, 103.53/255.0],
            std=[58.395/255.0, 57.12/255.0, 57.375/255.0]
        )

        self.transform = T.Compose([
            self.base_transform,
            self.normalize,
        ])
        
        # Label transform (resize + binarize)
        self.label_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
            BinaryLabelTransform()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        Returns:
            image_A (Tensor): shape [3, H, W]
            image_B (Tensor): shape [3, H, W]
            label   (Tensor): shape [1, H, W], values in {0,1}
            filename (str)  : image filename
        """
        filename = self.image_list[idx]
        
        # Load images
        image_A = Image.open(os.path.join(self.image_A_dir, filename)).convert('RGB')
        image_B = Image.open(os.path.join(self.image_B_dir, filename)).convert('RGB')
        label = Image.open(os.path.join(self.label_dir, filename)).convert('L')
        
        # Sanity check on label: must be {0, 255}
        label_np = np.array(label)
        unique_vals = np.unique(label_np)
        # Force any unknown values to 0
        if not set(unique_vals).issubset({0, 255}):
            label_np = np.where(np.isin(label_np, [0, 255]), label_np, 0)
            label = Image.fromarray(label_np.astype(np.uint8))
        
        # Apply transforms
        image_A = self.transform(image_A)
        image_B = self.transform(image_B)
        label = self.label_transform(label)
        
        return image_A, image_B, label, filename
