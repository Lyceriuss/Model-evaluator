import torch
import torchvision
import matplotlib
import tqdm

print("Matplotlib version:", matplotlib.__version__)
print("tqdm version:", tqdm.__version__)

# Test if CUDA is available
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available.")


import os
import json

#Count samples
def count_samples(root_dir):
    sample_count = 0
    invalid_samples = 0

    # Traverse the directories
    for month_dir in os.listdir(root_dir):
        month_path = os.path.join(root_dir, month_dir)
        if os.path.isdir(month_path):
            for day_dir in os.listdir(month_path):
                day_path = os.path.join(month_path, day_dir)
                if os.path.isdir(day_path):
                    files_in_day = os.listdir(day_path)
                    front_images = sorted([f for f in files_in_day if f.startswith("front") and f.endswith(".jpg")])
                    labels = sorted([f for f in files_in_day if f.endswith(".json")])

                    # Ensure equal number of images and labels
                    if len(front_images) == len(labels):
                        for front_image, label in zip(front_images, labels):
                            image_path = os.path.join(day_path, front_image)
                            label_path = os.path.join(day_path, label)

                            # Check if files exist
                            if os.path.isfile(image_path) and os.path.isfile(label_path):
                                # Validate JSON
                                try:
                                    with open(label_path, 'r') as f:
                                        json.load(f)
                                    sample_count += 1
                                except json.JSONDecodeError:
                                    invalid_samples += 1
                                    print(f"Invalid JSON in {label_path}")
                            else:
                                invalid_samples += 1
                                print(f"Missing image or label for {front_image} and {label}")
                    else:
                        print(f"Mismatched images and labels in {day_path}: {len(front_images)} images, {len(labels)} labels")
                        invalid_samples += min(len(front_images), len(labels))

    print(f"Total valid samples: {sample_count}")
    if invalid_samples > 0:
        print(f"Total invalid samples (missing files or invalid JSON): {invalid_samples}")

if __name__ == "__main__":
    root_dir = os.path.join(os.getcwd(), "data", "circular_fashion_v1_extracted")
    count_samples(root_dir)
