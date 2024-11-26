from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import numpy as np

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
])


class PatchDataset(Dataset):
    """
    Custom Dataset to handle patches with RGB images and mask pixel counts.
    """
    def __init__(self, patches):
        """
        Args:
            patches (list): List of numpy arrays, where each patch has 4 channels (RGB + mask).
        """
        self.images = []
        self.mask_pixels = []

        for patch in patches:
            # Process the image (first 3 channels for RGB)
            image = Image.fromarray(np.uint8(patch[:, :, :3]))
            self.images.append(transform(image))

            # Compute mask pixel count (4th channel as mask)
            mask = patch[:, :, 3]
            self.mask_pixels.append(np.sum(mask == 255))

        # Convert to tensors
        self.images = torch.stack(self.images)
        self.mask_pixels = torch.tensor(self.mask_pixels, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.mask_pixels[idx]