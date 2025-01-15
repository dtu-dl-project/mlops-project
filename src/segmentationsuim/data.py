import os
import gdown
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from torch.utils.data import random_split
import logging
from matplotlib.colors import ListedColormap

logger = logging.getLogger(__name__)


def download_dataset():
    """Download and extract the SUIM dataset if the "../data/raw" directory is empty."""
    url = "https://drive.google.com/file/d/1XFCe0DPhRxjJZmOtyxTIxrN247YLlLaQ/view"

    data_path_raw = "data/raw"

    # Ensure the directory exists
    os.makedirs(data_path_raw, exist_ok=True)

    # Check if the folder is empty (excluding .gitkeep)
    folder_content = [f for f in os.listdir(data_path_raw) if f != ".gitkeep"]

    if not folder_content:
        logger.info("Dataset folder empty, downloading dataset...")

        logger.info(f"Downloading dataset from {url}...")

        # Download the file using gdown
        downloaded = gdown.download(url, os.path.join(data_path_raw, "SUIM.tar.gz"), quiet=False, fuzzy=True)

        # Check if the file was downloaded successfully
        if downloaded is None:
            logger.error("Failed to download the dataset. Exiting...")
        else:
            logger.info(
                f"Downloaded dataset to {data_path_raw}. File size: {os.path.getsize(downloaded) / (1024 * 1024):.2f} MB"
            )

            # Extract the .tar.gz file
            try:
                logger.info("Extracting SUIM.tar.gz...")
                os.system(f"tar -xvf {os.path.join(data_path_raw, 'SUIM.tar.gz')} -C {data_path_raw}")
                logger.info("Extraction completed.")

                # Delete the .tar.gz file
                os.remove(os.path.join(data_path_raw, "SUIM.tar.gz"))
                logger.info("Deleted SUIM.tar.gz after extraction.")

                # Move contents of SUIM_fix to data/raw and delete SUIM_fix
                extracted_folder = os.path.join(data_path_raw, "SUIM_fix")
                if os.path.exists(extracted_folder):
                    for item in os.listdir(extracted_folder):
                        item_path = os.path.join(extracted_folder, item)
                        new_path = os.path.join(data_path_raw, item)
                        os.rename(item_path, new_path)  # Move files/folders up one level
                        logger.info(f"Moved {item} to {data_path_raw}.")

                    # Remove the SUIM_fix folder
                    os.rmdir(extracted_folder)
                    logger.info("Deleted SUIM_fix folder.")

            except Exception as e:
                logger.error(f"Failed to extract and organize files: {e}")

        logger.info("Dataset downloaded and organized successfully.")
    else:
        logger.info("The folder is not empty. Skipping download.")


def grayscale_to_class(mask):
    """
    Converts a grayscale mask image into a class map using a predefined grayscale-to-class mapping.

    :param mask: PIL.Image
        A grayscale mask image where each unique intensity value represents a distinct class.
    :return: np.ndarray
        A 2D array (H, W) of integer class IDs where each ID corresponds to a specific category.
    """
    # Define grayscale-to-class mapping
    grayscale_to_class = {
        0: 0,  # BW: Background/waterbody
        29: 1,  # HD: Human divers
        76: 2,  # PF: Aquatic plants and sea-grass
        105: 3,  # WR: Wrecks/ruins
        150: 4,  # RO: Robots/instruments
        179: 5,  # RI: Reefs/invertebrates
        226: 6,  # FV: Fish and vertebrates
        255: 7,  # SR: Sea-floor/rocks
    }

    # Convert the mapping into a NumPy array for vectorized lookup
    lut = np.zeros(256, dtype=np.uint8)  # LUT for all possible grayscale values
    for grayscale_value, class_id in grayscale_to_class.items():
        lut[grayscale_value] = class_id

    # Convert the PIL Image to a NumPy array
    mask_array = np.array(mask)

    # Use the LUT to map grayscale values to class IDs
    class_map = lut[mask_array]

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
    unique_classes : set
        A set of all unique class values found in the masks.

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
            Path to the dataset folder. It should be data/raw/train_val or data/raw/TEST.
            It should contain `images` and `masks` subfolders.
        :param image_transform: callable, optional
            Transformations to be applied to the images (e.g., resizing, normalization, etc.).
        :param mask_transform: callable, optional
            Transformations to be applied to the masks (e.g., resizing, etc.).
        """
        self.data_path = data_path
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.data = []

        counter = 0

        # Ensure the dataset folder exists
        os.makedirs(data_path, exist_ok=True)

        # Define paths for images and masks
        images_path = os.path.join(data_path, "images")
        masks_path = os.path.join(data_path, "masks")

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
                logger.debug(f"Found mask for {image_name}.")
                counter += 1
                self.data.append((image_path, mask_path))
            else:
                logger.info(f"Mask not found for {image_name}. Skipping...")
        logger.info(f"Dataset loaded with {counter} images and masks.")

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
        # Retrieve paths
        image_path, mask_path = self.data[idx]

        # Log paths for debugging
        logger.debug(f"Loading image from: {image_path}")
        logger.debug(f"Loading mask from: {mask_path}")

        # Load the image and mask as PIL images
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert the mask from RGB to a class map
        mask = grayscale_to_class(mask)
        logger.debug(f"Converted mask to class map with shape {mask.shape} and unique values: {np.unique(mask)}")

        # Apply transformations to the image
        if self.image_transform:
            image = self.image_transform(image)
            logger.debug(f"Image transformed with shape: {image.shape}")
        else:
            image = transforms.ToTensor()(image)
            logger.debug(f"Image converted to Tensor with shape: {image.shape}")

        # Apply transformations to the mask
        if self.mask_transform:
            # Convert mask to PIL Image for transformations
            mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L")
            mask = self.mask_transform(mask_pil)
            logger.debug(f"Mask transformed with shape: {mask.size if isinstance(mask, Image.Image) else mask.shape}")

            # Convert back to NumPy array and then to tensor
            mask_array = np.array(mask)
            mask_tensor = torch.tensor(mask_array, dtype=torch.uint8)
        else:
            # Convert directly to tensor
            mask_tensor = torch.tensor(mask, dtype=torch.uint8)

        # Log final mask properties
        logger.debug(f"Final mask tensor shape: {mask_tensor.shape}")
        logger.debug(f"Final mask tensor dtype: {mask_tensor.dtype}")
        logger.debug(f"Final mask tensor unique values: {torch.unique(mask_tensor)}")

        return image, mask_tensor


def save_processed_dataset(dataset, output_path):
    """
    Save the processed dataset (images and masks) into the specified output folder.

    :param dataset: The dataset object to process.
    :param output_path: The root path where the processed dataset will be saved.
                        It should be data/processed/train_val or data/processed/test.
    """
    images_dir = os.path.join(output_path, "images")
    masks_dir = os.path.join(output_path, "masks")

    # Create directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    logger.info(f"Saving processed data to {output_path}...")

    for idx in tqdm(range(len(dataset)), desc="Processing dataset"):
        # Load the image and mask from the dataset
        image, mask = dataset[idx]

        # Ensure the image tensor is in [0, 255] and of type uint8
        if image.dtype != torch.uint8:
            image = (image * 255).clamp(0, 255).to(torch.uint8)

        # Convert image to PIL format (RGB)
        image_pil = to_pil_image(image)

        # Convert mask tensor to uint8 for PIL compatibility
        if mask.dtype != torch.uint8:
            mask = mask.to(torch.uint8)

        logger.debug(f"Mask shape: {mask.shape}")
        logger.debug(f"Mask min: {mask.min()}, max: {mask.max()}")
        logger.debug(f"Mask unique values: {torch.unique(mask)}")
        logger.debug(f"Mask dtype: {mask.dtype}")

        # Convert mask to PIL format (Grayscale)
        mask_pil = Image.fromarray(mask.cpu().numpy(), mode="L")

        # Save image and mask as PNG files
        image_pil.save(os.path.join(images_dir, f"{idx:05d}.png"))
        mask_pil.save(os.path.join(masks_dir, f"{idx:05d}.png"))

        logger.info(f"Processed image and mask {idx:05d} saved.")

    logger.info(f"Processed data saved to {output_path}.")


class SUIMDatasetProcessed(Dataset):
    """
    A PyTorch Dataset for loading processed images and masks for the SUIM dataset.

    Attributes:
    ----------
    data_path : str
        Path to the dataset folder containing `images` and `masks` subfolders.
        It should be data/processed/train_val or data/processed/test.
    data : list of tuples
        A list of tuples where each tuple contains the paths of an image and its corresponding mask.

    Methods:
    -------
    __len__():
        Returns the total number of image-mask pairs in the dataset.
    __getitem__(idx):
        Loads and returns the image and mask at the specified index as PyTorch tensors.
        - Image: RGB format with shape (3, H, W).
        - Mask: Single-channel format with shape (H, W).
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
                logger.info(f"Mask not found for {image_name}. Skipping...")
        logger.info(f"Dataset loaded with {len(self.data)} images and masks.")

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
            - mask (torch.Tensor): Single-channel mask tensor with shape (H, W).
        """
        # Load image and mask as PIL Images
        image_path, mask_path = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Ensure mask is single-channel

        # Convert image to tensor with shape (3, H, W)
        image = transforms.ToTensor()(image)

        # Convert mask to tensor with shape (H, W) (no channel dimension for masks)
        mask = torch.tensor(np.array(mask), dtype=torch.uint8)  # Ensure mask stays uint8

        logger.debug(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        logger.debug(f"Image min: {image.min()}, max: {image.max()}")
        logger.debug(f"Mask min: {mask.min()}, max: {mask.max()}")
        logger.debug(f"Mask unique values: {torch.unique(mask)}")
        logger.debug(f"Mask dtype: {mask.dtype}")
        logger.debug(f"Image dtype: {image.dtype}")

        return image, mask


def get_dataloaders(data_path, use_processed, image_transform, mask_transform, batch_size, num_workers, split_ratio):
    train_path = os.path.join(data_path, "train_val")
    test_path = os.path.join(data_path, "test")

    if use_processed:
        logger.info("Using processed dataset...")
        train_dataset = SUIMDatasetProcessed(train_path)
        unique_classes = set()
        for idx in range(len(train_dataset)):
            image, mask = train_dataset[idx]
            unique_classes.update(torch.unique(mask).tolist())
        print(f"Unique classes found in the dataset: {unique_classes}")
        test_dataset = SUIMDatasetProcessed(test_path)
    else:
        logger.info("Using raw dataset...")
        logger.info("Processing the dataset...")
        train_dataset = SUIMDatasetRaw(train_path, image_transform, mask_transform)
        unique_classes = set()
        for idx in range(len(train_dataset)):
            image, mask = train_dataset[idx]
            unique_classes.update(torch.unique(mask).tolist())
        print(f"Unique classes found in the dataset: {unique_classes}")
        test_dataset = SUIMDatasetRaw(test_path, image_transform, mask_transform)

    train_size = int(split_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    logger.info(f"Train dataloader with {len(train_loader)} batches of dimension {batch_size}.")
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logger.info(f"Validation dataloader with {len(val_loader)} batches of dimension {batch_size}.")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    logger.info(f"Test dataloader with {len(test_loader)} batches of dimension {batch_size}.")

    return train_loader, val_loader, test_loader


def visualize_dataset(images, masks, batch_size=4):
    """
    Visualizes a batch of images and their corresponding masks.

    :param images: Tensor
        Batch of images with shape (B, C, H, W).
    :param masks: Tensor
        Batch of masks with shape (B, H, W).
    :param batch_size: int
        Number of images/masks to visualize.
    """

    class_colors = [
        (0, 0, 0),  # Class 0: Black (BW - Background/waterbody)
        (0, 0, 255),  # Class 1: Blue (HD - Human divers)
        (0, 255, 0),  # Class 2: Green (PF - Aquatic plants and sea-grass)
        (135, 206, 235),  # Class 3: Sky blue (WR - Wrecks/ruins)
        (255, 0, 0),  # Class 4: Red (RO - Robots/instruments)
        (255, 192, 203),  # Class 5: Pink (RI - Reefs/invertebrates)
        (255, 255, 0),  # Class 6: Yellow (FV - Fish and vertebrates)
        (255, 255, 255),  # Class 7: White (SR - Sea-floor/rocks)
    ]
    class_colors = np.array(class_colors) / 255.0  # Normalize to [0, 1]

    # Create a colormap
    fixed_cmap = ListedColormap(class_colors)
    fig, axes = plt.subplots(batch_size, 2, figsize=(8, batch_size * 2))
    for i in range(batch_size):
        axes[i, 0].imshow(images[i].permute(1, 2, 0))
        axes[i, 0].set_title(f"Image {i + 1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i], cmap=fixed_cmap, vmin=0, vmax=len(class_colors) - 1)
        axes[i, 1].set_title(f"Mask {i + 1}")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()


def main(use_processed=False):
    """
    The main function is used to test the dataset classes and functions.
    Set the `save_and_use_saved` variable to `True` to save the processed dataset and load it for analysis.
    """
    logging.basicConfig(level=logging.INFO)
    # Define the dataset path
    data_path = "data/raw"

    image_transform = transforms.Compose([transforms.Resize((572, 572)), transforms.ToTensor()])

    mask_transform = transforms.Compose([transforms.Resize((572, 572), interpolation=Image.NEAREST)])

    download_dataset()

    if use_processed:
        train_data_path = os.path.join(data_path, "train_val")
        test_data_path = os.path.join(data_path, "test")

        train_val_dataset = SUIMDatasetRaw(train_data_path, image_transform, mask_transform)
        test_dataset = SUIMDatasetRaw(test_data_path, image_transform, mask_transform)

        # Save the processed dataset
        processed_path_train = "data/processed/train_val"
        save_processed_dataset(train_val_dataset, processed_path_train)

        processed_path_test = "data/processed/test"
        save_processed_dataset(test_dataset, processed_path_test)

        # Load the processed dataset
        train_val_dataset_1 = SUIMDatasetProcessed(processed_path_train)

        random_idx = np.random.randint(0, len(train_val_dataset_1))

        # analyze the processed dataset
        image, mask = train_val_dataset_1[random_idx]
        logger.info(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        logger.info(f"Image min: {image.min()}, max: {image.max()}")
        logger.info(f"Mask min: {mask.min()}, max: {mask.max()}")
        logger.info(f"Mask unique values: {torch.unique(mask)}")
        logger.info(f"Mask dtype: {mask.dtype}")
        logger.info(f"Image dtype: {image.dtype}")

        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            data_path="data/processed",
            use_processed=use_processed,
            image_transform=None,
            mask_transform=None,
            batch_size=8,
            num_workers=4,
            split_ratio=0.8,
        )

        # analysis on the dataloader

        batch = next(iter(train_dataloader))

        images, masks = batch

        logger.info(f"Batch images shape: {images.shape}, Batch masks shape: {masks.shape}")
        logger.info(f"Batch images min: {images.min()}, max: {images.max()}")
        logger.info(f"Batch masks min: {masks.min()}, max: {masks.max()}")
        logger.info(f"Batch masks unique values: {torch.unique(masks)}")
        logger.info(f"Batch masks dtype: {masks.dtype}")
        logger.info(f"Batch images dtype: {images.dtype}")

        # Visualize the batch images and masks
        visualize_dataset(images, masks, 8)
    else:
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            data_path=data_path,
            use_processed=use_processed,
            image_transform=image_transform,
            mask_transform=mask_transform,
            batch_size=8,
            num_workers=4,
            split_ratio=0.8,
        )

        # analysis on the dataloader

        batch = next(iter(train_dataloader))

        images, masks = batch

        logger.info(f"Batch images shape: {images.shape}, Batch masks shape: {masks.shape}")
        logger.info(f"Batch images min: {images.min()}, max: {images.max()}")
        logger.info(f"Batch masks min: {masks.min()}, max: {masks.max()}")
        logger.info(f"Batch masks unique values: {torch.unique(masks)}")
        logger.info(f"Batch masks dtype: {masks.dtype}")
        logger.info(f"Batch images dtype: {images.dtype}")

        # Set up the figure
        visualize_dataset(images, masks, batch_size=8)


if __name__ == "__main__":
    main(use_processed=True)
