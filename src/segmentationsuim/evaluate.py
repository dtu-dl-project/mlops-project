from segmentationsuim.train import UNetModule, unet, Trans
from segmentationsuim.data import download_dataset, get_dataloaders
from omegaconf import DictConfig

import torch as T
import logging
import hydra
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchmetrics.segmentation import MeanIoU
from tqdm import tqdm

NUM_CLASSES = 8
DEVICE = T.device("cuda" if T.cuda.is_available() else "mps" if T.backends.mps.is_available() else "cpu")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model."""
    logger.info("Evaluating model...")

    logger.info(f"Loading model from {cfg.checkpoints.dirpath + cfg.checkpoints.filename}")
    if cfg.checkpoints.filename.startswith("unet"):
        model = UNetModule.load_from_checkpoint(cfg.checkpoints.dirpath + cfg.checkpoints.filename, unet=unet)
    elif cfg.checkpoints.filename.startswith("transformer"):
        model = Trans.load_from_checkpoint(cfg.checkpoints.dirpath + cfg.checkpoints.filename)
    else:
        raise ValueError('Invalid checkpoint filename. It should start either with "unet" or "transformer"')
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
        if cfg.checkpoints.filename.startswith("unet"):
            pred = model.unet(img)
            pred = T.argmax(pred, dim=1)
        elif cfg.checkpoints.filename.startswith("transformer"):
            img_proc = model.processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
            y_hat = model.model(img_proc)
            logits = y_hat.logits
            logits_resized = F.interpolate(logits, size=model.image_size, mode="bilinear", align_corners=False)
            pred = T.argmax(logits_resized, dim=1)
        else:
            raise ValueError('Invalid checkpoint filename. It should start either with "unet" or "transformer"')
        iou.update(pred.long(), target.long())

    iou_score = iou.compute()
    logger.info(f"Mean Intersection Over Unit: {iou_score}")


if __name__ == "__main__":
    evaluate()
