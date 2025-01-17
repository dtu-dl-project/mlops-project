from segmentationsuim.data import download_dataset
from segmentationsuim.data import grayscale_to_class
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
