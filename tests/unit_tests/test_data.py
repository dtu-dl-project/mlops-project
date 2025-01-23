from segmentationsuim.data import download_dataset
from segmentationsuim.data import grayscale_to_class
from segmentationsuim.data import SUIMDatasetRaw
from segmentationsuim.data import SUIMDatasetProcessed
from segmentationsuim.data import save_processed_dataset
from segmentationsuim.data import get_dataloaders
from torchvision import transforms
from PIL import Image
import numpy as np


def test_download_dataset():
    assert download_dataset is not None


def test_grayscale_to_class():
    """Test grayscale-to-class conversion."""

    # Create a dummy grayscale image
    grayscale_image = Image.fromarray(np.array([[0, 29, 76], [105, 150, 179]], dtype=np.uint8))
    class_map = grayscale_to_class(grayscale_image)
    expected_class_map = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    assert np.array_equal(class_map, expected_class_map), "Grayscale to class mapping failed."


def test_Dataset_class():
    path = "tests/test_data/raw"
    dataset = SUIMDatasetRaw(path)
    image, mask = dataset[0]
    assert dataset is not None
    assert len(dataset) == 3
    assert image.dim() == 3
    assert mask.dim() == 2


def test_download_and_load_processed():
    path_raw = "tests/test_data/raw"
    path_processed = "tests/test_data/processed/test"
    image_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.Resize((256, 256), interpolation=Image.NEAREST)])
    dataset = SUIMDatasetRaw(path_raw, image_transform=image_transform, mask_transform=mask_transform)
    save_processed_dataset(dataset, path_processed)
    dataset1 = SUIMDatasetProcessed(path_processed)
    image, mask = dataset[0]
    assert dataset1 is not None
    assert image.dim() == 3
    assert mask.dim() == 2


def test_dataloadings():
    path = "tests/test_data/processed"
    dataloaders = get_dataloaders(
        path,
        use_processed=True,
        image_transform=None,
        mask_transform=None,
        batch_size=1,
        num_workers=1,
        split_ratio=0.5,
    )
    assert dataloaders is not None
    assert len(dataloaders) == 3
    for dataloader in dataloaders:
        for images, masks in dataloader:
            assert images.dim() == 4
            assert masks.dim() == 3
            break
        break
