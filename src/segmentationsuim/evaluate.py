from segmentationsuim.model import UNet
from segmentationsuim.data import download_dataset, get_dataloaders
from omegaconf import DictConfig

import torch
import logging
import hydra
from torchvision import transforms
from PIL import Image
from torchmetrics.segmentation import MeanIoU
from tqdm import tqdm

NUM_CLASSES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger = logging.getLogger(__name__)


def index_to_one_hot(tensor, num_classes=8):
    """
    Transforms a tensor with shape [N, W, H] containing index encoding
    into a one-hot encoded tensor with shape [N, num_classes, W, H].

    Args:
        tensor (torch.Tensor): Input tensor of shape [N, W, H] with index encoding.
        num_classes (int): Number of classes for one-hot encoding (default is 8).

    Returns:
        torch.Tensor: One-hot encoded tensor of shape [N, num_classes, W, H].
    """
    if tensor.dim() != 3:
        raise ValueError("Input tensor must have shape [N, W, H]")

    # Extract dimensions
    N, W, H = tensor.shape

    # Perform one-hot encoding
    one_hot = torch.zeros((N, num_classes, W, H), dtype=torch.long, device=tensor.device)
    one_hot.scatter_(1, tensor.unsqueeze(1), 1)

    return one_hot


@hydra.main(version_base=None, config_path="../../", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model."""
    logger.info("Evaluating model...")

    model = UNet(3, 8).to(DEVICE)
    logger.info(f"Loading model from {cfg.model_checkpoint_path}")
    model.load_state_dict(torch.load(cfg.model_checkpoint_path))
    logger.info("Model loaded successfully.")

    download_dataset()
    data_path = "data/raw"

    image_transform = transforms.Compose([transforms.Resize((572, 572)), transforms.ToTensor()])

    mask_transform = transforms.Compose([transforms.Resize((572, 572), interpolation=Image.NEAREST)])

    _, _, test_loader = get_dataloaders(
        data_path=data_path,
        use_processed=False,
        image_transform=image_transform,
        mask_transform=mask_transform,
        batch_size=cfg.data_loader.batch_size,
        num_workers=cfg.data_loader.workers,
        split_ratio=cfg.data_loader.split_ratio,
    )

    model.eval()
    iou = MeanIoU(num_classes=NUM_CLASSES).to(DEVICE)
    for img, target in tqdm(test_loader, desc="Evaluating"):
        img, target = img.to(DEVICE), target.to(DEVICE)
        target = index_to_one_hot(target.long(), num_classes=NUM_CLASSES)
        y_pred = model(img).long()
        iou.update(y_pred, target)

    iou_score = iou.compute()
    logger.info(f"Mean Intersection Over Unit: {iou_score}")


if __name__ == "__main__":
    evaluate()
