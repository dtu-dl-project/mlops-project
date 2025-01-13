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

    This function processes the mask image as follows:
    1. Extracts the unique RGB colors present in the mask.
    2. Assigns a unique class ID to each unique RGB color.
    3. Creates a 2D class map where each pixel's value corresponds to the class ID of its RGB color.

    :param mask: PIL.Image
        An RGB mask image where each unique color represents a distinct class.
        Expected shape: (H, W, 3).
    :return: np.ndarray
        A 2D array (H, W) of integer class IDs where each ID corresponds to a specific RGB color.
    """
    # Convert the PIL Image to a NumPy array
    mask_array = np.array(mask)

    # Identify unique RGB colors in the mask
    unique_colors = np.unique(mask_array.reshape(-1, 3), axis=0)

    # Map each unique color to a unique class ID
    color_to_class = {tuple(color): i for i, color in enumerate(unique_colors)}

    # Initialize a 2D array to store class IDs
    class_map = np.zeros(mask_array.shape[:2], dtype=np.int64)

    # Map each pixel's color to its corresponding class ID
    for color, class_id in color_to_class.items():
        # Find all pixels in the mask that match the current color
        mask_class = np.all(mask_array == np.array(color), axis=-1)
        # Assign the class ID to the corresponding pixels in the class map
        class_map[mask_class] = class_id

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



    













    
>>>>>>> a9c93a4 (modify data.py)
