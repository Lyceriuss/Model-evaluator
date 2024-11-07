# test_import.py

import sys
import os

def run_tests():
    from data import DataLoaderManager
    import json

    print("Import Successful!")

    # Define the root directory for data
    root_dir = 'C:/Users/Delic/Desktop/Quincy-AI/Kodning/Resnet-50/data/circular_fashion_v1_extracted'

    # Verify that the root_dir exists
    if not os.path.isdir(root_dir):
        print(f"Data directory not found at: {root_dir}")
        sys.exit(1)

    # Initialize DataLoaderManager with required arguments
    try:
        data_manager = DataLoaderManager(
            root_dir=root_dir,
            test_split=0.1,
            batch_size=32,
            num_workers=4
        )
        print("DataLoaderManager instance created successfully.")
    except TypeError as e:
        print(f"Initialization Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        sys.exit(1)

    # Test the data loaders
    try:
        train_loader, test_loader, dataset = data_manager.load_data()
        print(f"Total samples in dataset: {len(dataset)}")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of testing batches: {len(test_loader)}")
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        sys.exit(1)

    # Fetch a single batch from the training loader
    try:
        for batch in train_loader:
            if batch is None:
                print("Received an empty batch. Skipping...")
                continue
            images, labels, image_paths = batch
            print(f"Batch Size: {images.size(0)}")
            print(f"Image Tensor Shape: {images.shape}")
            print(f"Labels: {labels}")
            print(f"Image Paths: {image_paths}")
            break  # Only fetch the first batch for testing
    except Exception as e:
        print(f"An error occurred while fetching a batch: {e}")
        sys.exit(1)

    print("All tests passed successfully!")

if __name__ == "__main__":
    # Add project root to sys.path
    project_root = 'C:/Users/Delic/Desktop/Quincy-AI/Kodning/Resnet-50'

    if project_root not in sys.path:
        sys.path.append(project_root)

    run_tests()
