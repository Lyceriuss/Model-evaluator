from config import transform, root_dir
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import ClothingDataset  # Relative import

class DataLoaderManager:
    def __init__(self, root_dir, test_split=0.2, batch_size=32, num_workers=4):
        """
        Initializes the DataLoaderManager with training and testing DataLoaders.

        Args:
            root_dir (str): Root directory of the dataset.
            test_split (float): Fraction of data to be used as test set.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses for data loading.
        """
        self.root_dir = root_dir
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = self.get_transforms()
        self.train_loader, self.test_loader, self.dataset = self.load_data()

    def get_transforms(self):
        """
        Defines the image transformations.

        Returns:
            torchvision.transforms.Compose: Composed transformations.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        """
        Loads the dataset and splits it into training and testing sets.

        Returns:
            tuple: Training DataLoader, Testing DataLoader, and the full dataset.
        """
        dataset = ClothingDataset(root_dir=self.root_dir, transform=self.transform)
        dataset_length = len(dataset)
        test_size = int(self.test_split * dataset_length)
        train_size = dataset_length - test_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ClothingDataset.custom_collate
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ClothingDataset.custom_collate
        )

        return train_loader, test_loader, dataset
