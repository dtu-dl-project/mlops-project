from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from segmentationsuim.model import UNet
from segmentationsuim.data import download_dataset, get_dataloaders

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

DEVICE = T.device("cuda" if T.cuda.is_available() else "mps" if T.backends.mps.is_available() else "cpu")


class UNetModule(L.LightningModule):
    def __init__(self, unet, lr, image_size: tuple[int, int]):
        super().__init__()
        self.unet = unet
        self.lr = lr
        self.image_size = image_size
        self.val_mean_iou = MeanIoU(num_classes=8, input_format="index")
        self.save_hyperparameters("lr", "image_size")

    def step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x)
        mask_transform = transforms.Compose([transforms.Resize(self.image_size, interpolation=Image.NEAREST)])
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
        self.val_mean_iou.update(preds, y.long())

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"val_loss": loss.item()})

        return loss

    def on_validation_epoch_end(self):
        mean_iou = self.val_mean_iou.compute()
        self.log("val_mean_iou", mean_iou, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"val_mean_iou": mean_iou.item()})
        self.val_mean_iou.reset()

    def configure_optimizers(self):
        return T.optim.Adam(self.parameters(), lr=self.lr)


class Trans(L.LightningModule):
    def __init__(self, lr, model_name, image_size):
        super().__init__()
        output_classes = 8

        if model_name not in ["nvidia/mit-b0", "nvidia/segformer-b0-finetuned-ade-512-512"]:
            raise ValueError(f"Invalid transformer model: {model_name}")

        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_name, num_labels=output_classes, ignore_mismatched_sizes=True
        )  # Ignore original head
        self.lr = lr
        self.image_size = image_size
        self.val_mean_iou = MeanIoU(num_classes=output_classes)
        self.save_hyperparameters("lr", "model_name", "image_size")

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
    logger.info("\nConfiguration file:\n%s", OmegaConf.to_yaml(cfg))
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

    img_size = cfg.image_transformations.image_size
    image_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)])

    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=data_path,
        use_processed=False,
        image_transform=image_transform,
        mask_transform=mask_transform,
        batch_size=cfg.data_loader.batch_size,
        num_workers=cfg.data_loader.workers,
        split_ratio=cfg.data_loader.split_ratio,
    )

    if cfg.training.model == "unet":
        model = UNetModule(unet, lr=cfg.training.optimizer.lr, image_size=(img_size, img_size))
    elif cfg.training.model == "transformer":
        model = Trans(
            lr=cfg.training.optimizer.lr, model_name=cfg.training.transformer_model, image_size=(img_size, img_size)
        )
    else:
        raise ValueError(f"Invalid model: {cfg.training.model}")

    model_name = cfg.training.model
    if model_name == "transformer":
        model_name += "_" + cfg.training.transformer_model[7:]

    logger.info(f"{cfg.training.transformer_model[7:]=}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoints.dirpath,
        save_top_k=3,
        monitor="val_mean_iou",
        mode="max",
        filename=f"{model_name}_bs={cfg.data_loader.batch_size}_img_size={img_size}_{{epoch}}-{{val_loss:.5f}}-{{val_mean_iou:.5f}}",
    )

    wandb_logger = WandbLogger(log_model="all")

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_callback],
        limit_train_batches=cfg.training.limits_batches.train,
        limit_val_batches=cfg.training.limits_batches.val,
        logger=wandb_logger,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if wandb_enabled:
        # Save artifacts if wandb is enabled
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            artifact = wandb.Artifact(
                name="best_model",  # Artifact name
                type="model",  # Artifact type
                description="Trained UNet model",  # Artifact description
                metadata=dict(cfg_dict),  # Optional metadata (e.g., training configuration)
            )
            artifact.add_file(best_model_path)  # Add the model file
            wandb.log_artifact(artifact)  # Log the artifact
            logger.info(f"Logged model artifact: {best_model_path}")


if __name__ == "__main__":
    load_dotenv()
    main()
