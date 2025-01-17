from omegaconf import DictConfig, OmegaConf
from segmentationsuim.model import UNet
from segmentationsuim.data import download_dataset, get_dataloaders
from segmentationsuim.utils import index_to_one_hot

from torchvision import transforms
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
import torch.nn.functional as F
import torch as T
from PIL import Image

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

import logging
import hydra
import wandb
import os
from dotenv import load_dotenv
from torchmetrics.segmentation import MeanIoU

logger = logging.getLogger(__name__)

unet = UNet(3, 8)  # 3 input channels, 8 output channels

IMAGE_SIZE = (572, 572)

DEVICE = T.device("cuda" if T.cuda.is_available() else "mps" if T.backends.mps.is_available() else "cpu")


class UNetModule(L.LightningModule):
    def __init__(self, unet, lr, image_size: tuple[int, int]):
        super().__init__()
        self.unet = unet
        self.lr = lr
        self.image_size = image_size
        self.save_hyperparameters(ignore=["unet"])
        self.val_mean_iou = MeanIoU(num_classes=8)

    def step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x)
        mask_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST)])
        y_hat = mask_transform(y_hat)
        loss = F.cross_entropy(y_hat, y.long())
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, batch_idx)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        wandb.log({"train_loss": loss.item()})

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        preds = T.argmax(y_hat, dim=1)
        one_hot_y = index_to_one_hot(y, num_classes=8)
        self.val_mean_iou.update(preds, one_hot_y)
        mean_iou = self.val_mean_iou.compute()

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mean_iou", mean_iou, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"val_loss": loss.item(), "val_mean_iou": mean_iou.item()})

        self.val_mean_iou.reset()
        return loss

    def configure_optimizers(self):
        return T.optim.Adam(self.parameters(), lr=self.lr)


class Trans(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        output_classes = 8

        self.model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_name, num_labels=output_classes, ignore_mismatched_sizes=True
        )  # Ignore original head
        self.lr = lr
        self.val_mean_iou = MeanIoU(num_classes=output_classes)

    def step(self, batch, batch_idx):
        x, y = batch  # x shape: Batch x 3 x Width x Height
        # y shape: Batch x Width x Height

        # Model output
        inputs = self.processor(images=x, return_tensors="pt").pixel_values.to(DEVICE)
        y_hat = self.model(inputs)
        logits = y_hat.logits
        logits_resized = F.interpolate(
            logits, size=y.shape[-2:], mode="bilinear", align_corners=False
        )  # Batch x Classes x Width x Height

        # Ground truth
        y = y.squeeze(1).long()  # Batch x Widht x Height

        # Loss
        loss = F.cross_entropy(logits_resized, y)
        return loss, logits_resized, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, batch_idx)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        wandb.log({"train_loss": loss.item()})

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, resized_labels = self.step(batch, batch_idx)
        preds = T.argmax(logits, dim=1)
        self.val_mean_iou.update(preds, resized_labels)
        mean_iou = self.val_mean_iou.compute()

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mean_iou", mean_iou, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"val_loss": loss.item(), "val_mean_iou": mean_iou.item()})

        self.val_mean_iou.reset()
        return loss

    def configure_optimizers(self):
        return T.optim.Adam(self.parameters(), lr=self.lr)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Configuration: {cfg}")
    logging.basicConfig(level=logging.INFO)

    wandb_api_key = os.getenv("WANDB_API_KEY")

    # Convert cfg to dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = {str(key): value for key, value in cfg_dict.items()}  # Ensure all keys are strings

    try:
        wandb.login(key=wandb_api_key)
        wandb.init(
            project="segmentationsuim",
            config=cfg_dict,
        )
        wandb_enabled = True
    except Exception as e:
        logger.warning(f"Failed to initialize WandB: {e}")
        wandb_enabled = False

    if not wandb_enabled:
        # Dummy function that is invoked when wandb is not enabled
        def wandb_log_stub(_):
            pass

        wandb.log = wandb_log_stub

    download_dataset()
    data_path = "data/raw"

    image_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST)])

    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=data_path,
        use_processed=False,
        image_transform=image_transform,
        mask_transform=mask_transform,
        batch_size=cfg.data_loader.batch_size,
        num_workers=cfg.data_loader.workers,
        split_ratio=cfg.data_loader.split_ratio,
    )

    # model = Trans(lr=cfg.training.optimizer.lr)
    model = UNetModule(unet, lr=cfg.training.optimizer.lr, image_size=IMAGE_SIZE)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoints.dirpath,
        save_top_k=3,
        monitor="val_mean_iou",
        filename="{epoch}-{val_loss:.5f}-{val_mean_iou:.5f}",
    )

    trainer = L.Trainer(max_epochs=cfg.training.max_epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    load_dotenv()
    main()
