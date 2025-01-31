import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A convolutional block that applies two consecutive convolution layers,
    each followed by batch normalization and ReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),  # Add dropout after the first activation
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),  # Add dropout after the second activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.
        :param x: Input tensor of shape (B, C, H, W).
        :return: Output tensor of shape (B, C, H, W).
        """
        return self.double_conv(x)


class Downscaling(nn.Module):
    """
    A downscaling block that applies max pooling followed by a convolutional block.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(Downscaling, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the downscaling block.
        :param x: Input tensor of shape (B, C, H, W).
        :return: Downscaled tensor of shape (B, C, H//2, W//2).
        """
        x = self.maxpool(x)
        return self.conv_block(x)


class Upscaling(nn.Module):
    """
    An upscaling block that applies transposed convolution for upsampling
    and concatenates the skip connection, followed by a convolutional block.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(Upscaling, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )  # in_channels // 2 since then we will concatenate the encoder part and so the number of channels will double
        self.conv_block = ConvBlock(in_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the upscaling block.
        :param x: Upsampled feature map of shape (B, C, H, W).
        :param skip: Skip connection from the encoder path of shape (B, C, H, W).
        :return: Output tensor of shape (B, C_out, H, W).
        """
        x = self.upsample(x)
        # print(f"Upscaling Output Shape: {x.shape}")
        x_height, x_width = x.size()[2], x.size()[3]
        skip_height, skip_width = skip.size()[2], skip.size()[3]
        diffY = skip_height - x_height
        diffX = skip_width - x_width
        # print(f"Skip Shape: {skip.shape}")
        # print(f"DiffY: {diffY}, DiffX: {diffX}")
        # Adjust the skip connection dimensions to match x
        if diffY > 0 or diffX > 0:
            # Crop the skip connection if it's larger than x
            skip = skip[
                :, :, diffY // 2 : skip_height - (diffY - diffY // 2), diffX // 2 : skip_width - (diffX - diffX // 2)
            ]
        elif diffY < 0 or diffX < 0:
            # Pad the skip connection if it's smaller than x
            skip = F.pad(skip, [-diffX // 2, -diffX + (-diffX // 2), -diffY // 2, -diffY + (-diffY // 2)])

        # print(f"Adjusted Skip Shape: {skip.shape}")
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class UNet(nn.Module):
    """
    A U-Net model implementation for image segmentation.
    """

    def __init__(
        self,
        num_input_channels: int = 1,
        num_output_classes: int = 1,
        base_feature_size: int = 64,
        dropout: float = 0.5,
    ):
        """
        Initialize the U-Net model.
        :param num_input_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        :param num_output_classes: Number of output classes for segmentation.
        :param base_feature_size: Number of features in the first layer, doubled in each downscaling.
        :param dropout: Dropout probability for regularization.
        """
        super(UNet, self).__init__()
        self.initial_block = ConvBlock(num_input_channels, base_feature_size, dropout)
        self.encoder1 = Downscaling(base_feature_size, base_feature_size * 2, dropout)
        self.encoder2 = Downscaling(base_feature_size * 2, base_feature_size * 4, dropout)
        self.encoder3 = Downscaling(base_feature_size * 4, base_feature_size * 8, dropout)
        self.encoder4 = Downscaling(base_feature_size * 8, base_feature_size * 16, dropout)

        self.decoder1 = Upscaling(base_feature_size * 16, base_feature_size * 8, dropout)
        self.decoder2 = Upscaling(base_feature_size * 8, base_feature_size * 4, dropout)
        self.decoder3 = Upscaling(base_feature_size * 4, base_feature_size * 2, dropout)
        self.decoder4 = Upscaling(base_feature_size * 2, base_feature_size, dropout)

        self.final_block = nn.Conv2d(base_feature_size, num_output_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net model.
        :param x: Input tensor of shape (B, C, H, W).
        :return: Output tensor of shape (B, num_output_classes, H, W).
        """
        # Encoder path
        x1 = self.initial_block(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        # Decoder path
        x = self.decoder1(x5, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)

        # Final segmentation map
        return self.final_block(x)

    def print_summary(self):
        """
        Print the model summary.
        """
        print(self)
        print("Trainable parameters: ", sum(p.numel() for p in self.parameters()))


if __name__ == "__main__":
    # Create a U-Net model
    model = UNet(num_input_channels=3, num_output_classes=10, base_feature_size=64, dropout=0.5)
    model.print_summary()
