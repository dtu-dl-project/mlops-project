import os
import zipfile
import gdown
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from PIL import Image
from torch.utils.data import random_split
import time



def download_dataset():
    """ SUIM Dataset is downloaded if the "../data/raw" directory is empty."""
    test_url = "https://drive.google.com/file/d/1diN3tNe2nR1eV3Px4gqlp6wp3XuLBwDy/view?usp=drive_link"
    train_url = "https://drive.google.com/file/d/1YWjUODQWwQ3_vKSytqVdF4recqBOEe72/view?usp=drive_link"

    test_ID = test_url.split("/d/")[1].split("/view")[0]
    train_ID = train_url.split("/d/")[1].split("/view")[0]

    data_path_raw = "data/raw"

    files = [
        {"id": test_ID, "name": "test.zip"},
        {"id": train_ID, "name": "train_val.zip"},
    ]

    # Ensure the directory exists
    os.makedirs(data_path_raw, exist_ok=True)

    # Check if the folder is empty (excluding .gitkeep)
    folder_content = [f for f in os.listdir(data_path_raw) if f != ".gitkeep"]

    if not folder_content:
        print("Dataset folder empty, downloading dataset...")

        for file in files:
            file_url = f"https://drive.google.com/uc?id={file['id']}"
            output_path = os.path.join(data_path_raw, file["name"])

            # Download the file
            if not os.path.exists(output_path):
                print(f"Downloading {file['name']} from {file_url}...")
                downloaded = gdown.download(file_url, output_path, quiet=False, fuzzy=True)

                # Check if the file was downloaded successfully
                if downloaded is None:
                    print(f"Failed to download {file['name']}. Skipping...")
                    continue

                print(
                    f"Downloaded {file['name']} to {output_path}. File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB"
                )

                # Extract if it's a zip file
                if output_path.endswith(".zip"):
                    print(f"Extracting {output_path}...")
                    try:
                        with zipfile.ZipFile(output_path, "r") as zip_ref:
                            for file_name in zip_ref.namelist():
                                zip_ref.extract(file_name, data_path_raw)
                                print(f"Extracted: {file_name}")
                        print(f"Extracted {file['name']} to {data_path_raw}.")
                    except zipfile.BadZipFile:
                        print(f"Failed to extract {file['name']}. The file may be corrupted.")

                    # Delete the zip file after extraction
                    os.remove(output_path)
                    print(f"Deleted {output_path} after extraction.")
            else:
                print(f"{file['name']} already exists, skipping download.")
        print("Dataset downloaded and extracted successfully.")
    else:
        print("The folder is not empty. Skipping download.")


def rgb_to_class(mask):
    """
    Converts an RGB mask image into a class map where each unique RGB color in the image 
    corresponds to a unique class ID.

    :param mask: PIL.Image
        An RGB mask image where each unique color represents a distinct class.
        Expected shape: (H, W, 3).
    :return: np.ndarray
        A 2D array (H, W) of integer class IDs where each ID corresponds to a specific RGB color.
    """
    # Convert the PIL Image to a NumPy array
    mask_array = np.array(mask)

    # Start timing for unique color extraction
    start_time = time.time()
    # Identify unique RGB colors in the mask
    unique_colors, inverse_indices = np.unique(
        mask_array.reshape(-1, 3), axis=0, return_inverse=True
    )
    end_time = time.time()
    print(f"Time for extracting unique colors: {end_time - start_time:.4f} seconds")

    # Create a class map using the inverse indices
    start_time = time.time()
    class_map = inverse_indices.reshape(mask_array.shape[:2])
    end_time = time.time()
    print(f"Time for creating class map: {end_time - start_time:.4f} seconds")

    if (end_time - start_time) > 10:
        # Visualize the mask
        plt.imshow(class_map)
        plt.show()

    return class_map
    
class SUIMDatasetRaw(Dataset):
    """
    A PyTorch Dataset class to load raw images and corresponding masks for the SUIM dataset.

    The class performs the following:
    - Initializes the dataset by checking the existence of `images` and `masks` folders.
    - Loads image-mask pairs if the corresponding mask exists for each image.
    - Applies optional transformations to both images and masks.
    - Converts masks from RGB to class maps with integer class IDs.

    Attributes:
    ----------
    data_path : str
        The path to the root directory containing the `images` and `masks` folders.
    image_transform : callable, optional
        Transformation to be applied to the images (e.g., resizing, normalization, etc.).
    mask_transform : callable, optional
        Transformation to be applied to the masks (e.g., resizing, etc.).
    data : list of tuples
        A list of tuples where each tuple contains the paths of an image and its corresponding mask.

    Methods:
    -------
    __len__():
        Returns the number of image-mask pairs in the dataset.
    __getitem__(idx):
        Returns the transformed image and mask for the given index.
    """
    def __init__(self, data_path: str, image_transform=None, mask_transform=None):
        """
        Initialize the SUIMDatasetRaw.

        :param data_path: str
            Path to the dataset folder. It should contain `images` and `masks` subfolders.
        :param image_transform: callable, optional
            Transformations to be applied to the images (e.g., resizing, normalization, etc.).
        :param mask_transform: callable, optional
            Transformations to be applied to the masks (e.g., resizing, etc.).
        """
        self.data_path = data_path
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.data = []

        # Ensure the dataset folder exists
        os.makedirs(data_path, exist_ok=True)

        # Define paths for images and masks
        images_path = os.path.join(data_path, "images")
        masks_path = os.path.join(data_path, "masks")
        count = 0

        # Check if the images and masks folders exist
        if not os.path.exists(images_path) or not os.path.exists(masks_path):
            raise ValueError(f"Images or masks folder not found in {data_path}.")

        # Load image-mask pairs
        for image_name in os.listdir(images_path):
            image_path = os.path.join(images_path, image_name)
            # Extract the base name without file extension
            image_name = image_name.split(".")[0]
            mask_path = os.path.join(masks_path, f"{image_name}.bmp")

            if os.path.exists(mask_path):
                print(f"Found mask for {image_name}.")
                count += 1
                self.data.append((image_path, mask_path))
            else:
                print(f"Mask not found for {image_name}. Skipping...")
        print(f"Loaded {count} images and masks.")

    def __len__(self):
        """
        Return the number of image-mask pairs in the dataset.

        :return: int
            The total number of pairs.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding mask by index.

        The function performs the following:
        - Loads the image as an RGB PIL image.
        - Loads the mask as an RGB PIL image.
        - Converts the mask to a class map (categorical class IDs).
        - Applies transformations to both the image and the mask.

        :param idx: int
            Index of the image-mask pair to retrieve.
        :return: tuple
            A tuple containing the transformed image (Tensor) and mask (Tensor with class IDs).
        """
        image_path, mask_path = self.data[idx]

        # Load image and mask as PIL Images
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert mask to a class map (categorical class IDs)
        mask = rgb_to_class(mask)

        # Apply transformations to the image
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Apply transformations to the mask
        # Apply transformations to the mask
        if self.mask_transform:
            # Convert the mask to a PIL Image for resizing and other transformations
            mask = Image.fromarray(mask.astype(np.uint8), mode="L")  # Ensure grayscale mod

            # Apply the transformation (e.g., resizing)
            mask = self.mask_transform(mask)

            # Check if the transformation returned a tensor
            if isinstance(mask, torch.Tensor):
                # Ensure mask is of type long and has a single channel
                #visualize the mask
                print(mask.shape)
            else:
                # If still a PIL image, convert it to a tensor
                mask = transforms.ToTensor()(mask)
        else:
            # Without transformations, convert the mask to a long tensor
            mask = torch.tensor(mask, dtype=torch.int32).unsqueeze(0)

        # Debug shapes after applying transforms
        print(f"After transformation - Image shape: {image.shape}, Mask shape: {mask.shape}")

        return image, mask


def save_processed_dataset(dataset, output_path):
    """
    Save the processed dataset (images and masks) into the specified output folder.
    
    :param dataset: The dataset object to process.
    :param output_path: The root path where the processed dataset will be saved.
    """
    images_dir = os.path.join(output_path, "images")
    masks_dir = os.path.join(output_path, "masks")

    # Create directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    print(f"Saving processed data to {output_path}...")

    for idx in tqdm(range(len(dataset)), desc="Processing dataset"):
        # Load the image and mask from the dataset
        image, mask = dataset[idx]

        # Ensure the image tensor is in [0, 255] and of type uint8
        if image.dtype != torch.uint8:
            image = (image * 255).clamp(0, 255).to(torch.uint8)

        # Convert image to PIL format (RGB)
        image_pil = to_pil_image(image)

        # Convert mask to PIL format (Grayscale)
        mask_pil = to_pil_image(mask)

        # Save image and mask as PNG files
        image_pil.save(os.path.join(images_dir, f"{idx:05d}.png"))
        mask_pil.save(os.path.join(masks_dir, f"{idx:05d}.png"))

        print(f"Processed image and mask {idx:05d} saved.")

    print(f"Processed data saved to {output_path}.")

class SUIMDatasetProcessed(Dataset):
    """
    A PyTorch Dataset for loading processed images and masks for the SUIM dataset.

    Attributes:
    ----------
    data_path : str
        Path to the dataset folder containing `images` and `masks` subfolders.
    data : list of tuples
        A list of tuples where each tuple contains the paths of an image and its corresponding mask.

    Methods:
    -------
    __len__():
        Returns the total number of image-mask pairs in the dataset.
    __getitem__(idx):
        Loads and returns the image and mask at the specified index as PyTorch tensors.
        - Image: RGB format with shape (3, H, W).
        - Mask: Single-channel format with shape (1, H, W).
    """
    def __init__(self, data_path: str):
        """
        Initialize the SUIM dataset.
        
        :param data_path: str
            Path to the dataset folder. It should contain `images` and `masks` subfolders.
        """
        self.data_path = data_path
        self.data = []

        os.makedirs(data_path, exist_ok=True)

        images_path = os.path.join(data_path, "images")
        masks_path = os.path.join(data_path, "masks")

        # Check if the images and masks folders exist
        if not os.path.exists(images_path) or not os.path.exists(masks_path):
            raise ValueError(f"Images or masks folder not found in {data_path}.")

        # Load image and mask paths
        for image_name in os.listdir(images_path):
            image_path = os.path.join(images_path, image_name)
            mask_path = os.path.join(masks_path, image_name)

            if os.path.exists(mask_path):
                self.data.append((image_path, mask_path))
            else:
                print(f"Mask not found for {image_name}. Skipping...")
        print(f"Loaded {len(self.data)} images and masks.")

    def __len__(self):
        """
        Return the number of image-mask pairs in the dataset.

        :return: int
            The total number of pairs.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Load and return an image and its corresponding mask by index.

        :param idx: int
            Index of the image-mask pair to retrieve.
        :return: tuple
            A tuple containing:
            - image (torch.Tensor): RGB image tensor with shape (3, H, W).
            - mask (torch.Tensor): Single-channel mask tensor with shape (1, H, W).
        """
        # Load image and mask as PIL Images
        image_path, mask_path = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Ensure mask is single-channel

        # Convert image to tensor with shape (3, H, W)
        image = transforms.ToTensor()(image)

        # Convert mask to tensor with shape (1, H, W)
        mask = transforms.ToTensor()(mask)

        return image, mask
    
def get_dataloaders(data_path, use_processed, image_transform, mask_transform, batch_size, num_workers, split_ratio):

    train_path = os.path.join(data_path, "train_val")
    test_path = os.path.join(data_path, "test")

    if use_processed:
        train_dataset = SUIMDatasetProcessed(train_path)
        test_dataset = SUIMDatasetProcessed(test_path)
    else:
        train_dataset = SUIMDatasetRaw(train_path, image_transform, mask_transform)
        test_dataset = SUIMDatasetRaw(test_path, image_transform, mask_transform)

    train_size = int(split_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
        
def main():
    # Define the dataset path
    data_path = "data/raw"

    train_val_path = os.path.join(data_path, "train_val")
    test_path = os.path.join(data_path, "test")

    image_transform = transforms.Compose([
        transforms.Resize((572, 572)),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((572, 572), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    train_loader, val_loader, test_loader = get_dataloaders(data_path=data_path, use_processed=False, image_transform=image_transform, mask_transform=mask_transform, batch_size=32, num_workers=4, split_ratio=0.8)

    # Visualize the first batch of images and masks
    images, masks = next(iter(train_loader))
    print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")

    fig, axes = plt.subplots(4, 2, figsize=(12, 24))
    for i in range(4):
        image = transforms.ToPILImage()(images[i])
        mask = transforms.ToPILImage()(masks[i])
        axes[i, 0].imshow(image)
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import cProfile
    download_dataset()
    # save output to a file
    cProfile.run("main()", sort="cumtime", filename="output_sorted.prof")
