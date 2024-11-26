import timm
import torch
from torch.utils.data import DataLoader
from .patch_dataset import PatchDataset


class Detector:
    def __init__(self, model_name="resnet50", num_classes=10, model_path='models/resnet50_eurosat.pth'):
        """
        Initialize the detector with a specified model.

        Args:
            model_name (str): The name of the model to use.
            num_classes (int): The number of classes for the classification task.
            model_path (str): Path to the pretrained model weights.
        """
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()  


    def predict(self, patches, batch_size=8, device='cuda'):
        """
        Predict the class of each patch and compute pixel counts per class.

        Args:
            patches (list): List of image patches, each with 4 channels (RGB + mask).
            batch_size (int): Batch size for prediction.
            device (str): Device to run the model on ('cuda' or 'cpu').

        Returns:
            class_pixel_counts (list): Total number of pixels (with mask applied) per class.
        """
        # Create a dataset and DataLoader for the patches
        dataset = PatchDataset(patches)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Move model to the desired device
        self.model.to(device)

        # Initialize pixel count accumulator
        class_pixel_counts = [0] * self.model.num_classes

        # Predict in batches
        with torch.no_grad():
            for images, mask_pixels in dataloader:
                images = images.to(device)
                mask_pixels = mask_pixels.to(device)

                # Forward pass through the model
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                # Accumulate pixel counts for each predicted class
                for i, pred_class in enumerate(preds):
                    class_pixel_counts[pred_class.item()] += mask_pixels[i].item()

        return class_pixel_counts