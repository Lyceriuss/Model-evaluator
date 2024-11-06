# my_package/dataset.py

import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

class ClothingDataset(Dataset):
    def __init__(self, root_dir, transform=None, sample_size=None, label_key='type'):
        """
        Initializes the dataset by collecting image paths and labels.

        Args:
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): Transform to be applied to images.
            sample_size (int, optional): Number of samples to use.
            label_key (str): Key to extract the label from JSON files.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.label_key = label_key

        # Collect image and label file paths
        self.image_files, self.label_files = self._collect_files(sample_size)

        # Collect all unique labels and create mappings
        self.labels_set = self._collect_labels()
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(self.labels_set))}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}


    def _collect_files(self, sample_size):
        image_files = []
        label_files = []
        count = 0

        # Traverse directories to collect front images and labels
        for month_dir in os.listdir(self.root_dir):
            month_path = os.path.join(self.root_dir, month_dir)
            if os.path.isdir(month_path):
                days_in_month = os.listdir(month_path)
                for day in days_in_month:
                    day_path = os.path.join(month_path, day)
                    if os.path.isdir(day_path):
                        files_in_day = os.listdir(day_path)
                        front_images = sorted([f for f in files_in_day if f.startswith("front") and f.endswith(".jpg")])
                        labels = sorted([f for f in files_in_day if f.endswith(".json")])

                        # Ensure equal number of images and labels
                        if len(front_images) == len(labels):
                            for front_image, label in zip(front_images, labels):
                                image_path = os.path.join(day_path, front_image)
                                label_path = os.path.join(day_path, label)

                                # Validate JSON
                                try:
                                    with open(label_path, 'r') as f:
                                        json.load(f)
                                except json.JSONDecodeError as e:
                                    print(f"Skipping {label_path} due to JSONDecodeError: {e}")
                                    continue

                                image_files.append(image_path)
                                label_files.append(label_path)
                                count += 1

                                if sample_size and count >= sample_size:
                                    return image_files, label_files
        return image_files, label_files
    
    @staticmethod
    def custom_collate(batch):
        # Filter out None samples
        batch = [item for item in batch if item is not None]
        if not batch:
            return None  # or raise an exception
        transposed = list(zip(*batch))
        images = default_collate(transposed[0])
        labels = default_collate(transposed[1])
        image_paths = transposed[2]  # This is a list of image paths
        return images, labels, image_paths

    def _collect_labels(self):
        labels_set = set()
        for label_file in self.label_files:
            try:
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                    clothing_label = label_data.get(self.label_key, None)
                if clothing_label:
                    labels_set.add(clothing_label)
                else:
                    print(f"Warning: '{self.label_key}' not found in {label_file}")
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
        return labels_set

    def __len__(self):
        return len(self.image_files)
    

    def clean_label(label):
        if label:
            label = label.strip().lower()
            # Correct common typos or variations
            corrections = {
                'jacker': 'jacket',
                'night gown': 'nightgown',
                'tank top ': 'tank top',
                'sweater': 'sweater',  # In case of capitalization differences
                # Add more corrections as needed
            }
            label = corrections.get(label, label)
        return label


    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        # Open image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        try:
            with open(label_path, 'r') as f:
                label_data = json.load(f)
            clothing_label = label_data.get(self.label_key, None)
            if clothing_label is None:
                print(f"Missing or invalid '{self.label_key}' in {label_path}")
                raise IndexError  # Skip this sample
            label_index = self.label_to_index[clothing_label]
        except Exception as e:
            print(f"Error processing label file {label_path}: {e}")
            raise IndexError  # Skip this sample

        return image, label_index, image_path  # Image path returned for further logging
    
    
