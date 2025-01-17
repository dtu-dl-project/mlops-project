from segmentationsuim.train import UNetModule, unet
from segmentationsuim.data import download_dataset, get_dataloaders
from segmentationsuim.utils import index_to_one_hot
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


@hydra.main(version_base=None, config_path="../../", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model."""
    logger.info("Evaluating model...")

    logger.info(f"Loading model from {cfg.checkpoints.dirpath + cfg.checkpoints.filename}")
    model = UNetModule.load_from_checkpoint(cfg.checkpoints.dirpath + cfg.checkpoints.filename, unet=unet)
    logger.info("Model loaded successfully.")

    download_dataset()
    data_path = "data/raw"

    image_transform = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor()])

    mask_transform = transforms.Compose([transforms.Resize(model.image_size, interpolation=Image.NEAREST)])

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
        y_pred = model.unet(img).long()
        iou.update(y_pred, target)

    iou_score = iou.compute()
    logger.info(f"Mean Intersection Over Unit: {iou_score}")


if __name__ == "__main__":
    evaluate()
