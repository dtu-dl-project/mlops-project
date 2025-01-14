from segmentationsuim.model import UNet
from segmentationsuim.data import download_dataset, get_dataloaders

from torchvision import transforms
import lightning as L
import torch.nn.functional as F
import torch as T
from PIL import Image
import logging

logger = logging.getLogger(__name__)


unet = UNet(3, 8)  # 3 input channels, 1 output channel (not true)


class UNetModule(L.LightningModule):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x)

        y = y.squeeze(1)  # remove channel dimension to make cross_entropy happy

        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return T.optim.Adam(self.parameters(), lr=1e-3)


def main():
    logging.basicConfig(level=logging.INFO)

    download_dataset()
    data_path = "data/raw"

    image_transform = transforms.Compose([transforms.Resize((572, 572)), transforms.ToTensor()])

    mask_transform = transforms.Compose(
        [transforms.Resize((572, 572), interpolation=Image.NEAREST), transforms.ToTensor()]
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=data_path,
        use_processed=False,
        image_transform=image_transform,
        mask_transform=mask_transform,
        batch_size=4,
        num_workers=4,
        split_ratio=0.8,
    )

    model = UNetModule(unet)
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
