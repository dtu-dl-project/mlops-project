from segmentationsuim.train import UNetModule, Trans, unet
from segmentationsuim.data import download_dataset, get_dataloaders
from omegaconf import DictConfig

import matplotlib.pyplot as plt
import torch
import hydra
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def visualize(cfg: DictConfig) -> None:
    """Visualize model predictions."""

    if cfg.checkpoints.filename.startswith("unet"):
        model = UNetModule.load_from_checkpoint(cfg.checkpoints.dirpath + cfg.checkpoints.filename, unet=unet)
    elif cfg.checkpoints.filename.startswith("transformer"):
        model = Trans.load_from_checkpoint(cfg.checkpoints.dirpath + cfg.checkpoints.filename)
    else:
        raise ValueError('Invalid checkpoint filename. It should start either with "unet" or "transformer"')
    model.eval()

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

    images, targets, predictions = [], [], []

    # Collect images, targets, and predictions
    with torch.inference_mode():
        for img, target in tqdm(test_loader, desc="Evaluating"):
            img, target = img.to(DEVICE), target.to(DEVICE)

            if cfg.checkpoints.filename.startswith("unet"):
                pred = model.unet(img)
            elif cfg.checkpoints.filename.startswith("transformer"):
                img_proc = model.processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
                y_hat = model.model(img_proc)
                logits = y_hat.logits
                pred = logits.argmax(dim=1)
            else:
                raise ValueError('Invalid checkpoint filename. It should start either with "unet" or "transformer"')

            images.append(img.cpu())
            targets.append(target.cpu())
            predictions.append(pred.cpu())

    # Concatenate tensors after batching
    images = torch.cat(images).numpy()
    targets = torch.cat(targets).numpy()
    predictions = torch.cat(predictions).numpy()

    # Calculate the total number of images
    num_images = 5

    # Determine the grid size for the plots
    cols = 3  # One column each for image, target, and prediction
    rows = num_images  # One row per example

    # Set up the figure
    plt.figure(figsize=(15, 5 * rows))

    # Plot each image, target, and prediction
    for i in range(num_images):
        # Plot the original image
        plt.subplot(rows, cols, i * cols + 1)
        plt.imshow(images[i].transpose(1, 2, 0))
        plt.axis("off")
        plt.title(f"Image {i + 1}")

        # Plot the target
        plt.subplot(rows, cols, i * cols + 2)
        plt.imshow(targets[i].squeeze(), cmap="tab20", vmin=0, vmax=19)
        plt.axis("off")
        plt.title(f"Target {i + 1}")

        # Plot the prediction
        plt.subplot(rows, cols, i * cols + 3)

        if cfg.checkpoints.filename.startswith("unet"):
            plt.imshow(predictions[i].argmax(0), cmap="tab20", vmin=0, vmax=19)
        elif cfg.checkpoints.filename.startswith("transformer"):
            plt.imshow(predictions[i], cmap="tab20", vmin=0, vmax=19)
        else:
            raise ValueError('Invalid checkpoint filename. It should start either with "unet" or "transformer"')

        plt.axis("off")
        plt.title(f"Prediction {i + 1}")

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the figure
    dot_index = cfg.checkpoints.filename.rfind(".")
    filename = cfg.checkpoints.filename[:dot_index] if dot_index != -1 else cfg.checkpoints.filename
    plt.savefig(f"reports/figures/{filename}.png")

    # Optionally show the plot
    plt.show()


if __name__ == "__main__":
    visualize()
