from segmentationsuim.train import UNetModule, unet, Trans
from segmentationsuim.data import download_dataset, get_dataloaders
from omegaconf import DictConfig

import torch as T
import logging
import hydra
import sys
from pathlib import Path
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchmetrics.segmentation import MeanIoU
from tqdm import tqdm
import onnxruntime as rt
import os
import numpy as np

NUM_CLASSES = 8
DEVICE = T.device("cuda" if T.cuda.is_available() else "mps" if T.backends.mps.is_available() else "cpu")
logger = logging.getLogger(__name__)


def modify_file_extension(file_name: str, new_extension: str) -> str:
    """
    Modify the extension of a given file name.

    :param file_name: Original file name (with extension).
    :param new_extension: New extension to be added (with or without the leading dot).
    :return: File name with the new extension.
    """
    # Remove the current extension
    base_name = os.path.splitext(file_name)[0]

    # Ensure the new extension starts with a dot
    if not new_extension.startswith("."):
        new_extension = "." + new_extension

    # Return the file name with the new extension
    return base_name + new_extension


@hydra.main(version_base=None, config_path=None, config_name=None)
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

    # peak at the first image in the test_loader
    img, _ = next(iter(test_loader))

    # Export model to ONNX
    onnx_path = Path("models") / modify_file_extension(cfg.checkpoints.filename, "onnx")
    T.onnx.export(model, img, onnx_path, verbose=True)
    logger.info(f"Model stored as ONNX file: {onnx_path}")
    model.to_onnx(
        file_path=onnx_path,
        input_sample=img,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Load ONNX model
    ort_session = rt.InferenceSession(onnx_path)
    input_names = [i.name for i in ort_session.get_inputs()]
    output_names = [i.name for i in ort_session.get_outputs()]

    # Restart from first element
    test_loader = iter(test_loader)

    for img, target in tqdm(test_loader, desc="Evaluating"):
        img, target = img.to(DEVICE), target.to(DEVICE)

        # Preprocess input for ONNX model
        img_np = img.cpu().numpy().astype(np.float32)  # Convert to NumPy
        batch = {input_names[0]: img_np}  # Prepare batch input

        # Run inference
        ort_outs = ort_session.run(output_names, batch)

        # Process ONNX outputs
        logits = T.tensor(ort_outs[0])  # Convert output to PyTorch tensor
        logits = logits.to(DEVICE)

        # Resize and take argmax if necessary
        logits_resized = F.interpolate(logits, size=img.shape[2:], mode="bilinear", align_corners=False)
        pred = T.argmax(logits_resized, dim=1)

        # Update IoU metric
        iou.update(pred.long(), target.long())

    # Compute and log IoU
    iou_score = iou.compute()
    logger.info(f"Mean Intersection Over Unit: {iou_score}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 2:
        # Use the provided path from command-line arguments
        path = Path(sys.argv[1])
        config_path = "../../" + str(path.parent)
        config_name = str(path.name)

        ckpt_path = Path(sys.argv[2])
        ckpt_dir = ckpt_path.parent
        ckpt_name = ckpt_path.name
    else:
        raise FileNotFoundError(
            "Expected two arguments: (1) path to the config file and (2) path to the model checkpoint file. Please provide both paths."
        )

    hydra.initialize(config_path=config_path)
    cfg = hydra.compose(config_name=config_name)
    cfg.checkpoints.dirpath = str(ckpt_dir) + "/"
    cfg.checkpoints.filename = ckpt_name
    evaluate(cfg)
