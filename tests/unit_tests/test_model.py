import torch
from segmentationsuim.model import ConvBlock, Downscaling, Upscaling, UNet


def test_conv_block():
    """Test ConvBlock for correct output dimensions."""
    block = ConvBlock(in_channels=3, out_channels=64, dropout=0.1)
    x = torch.randn(2, 3, 572, 572)  # Batch of 2, 3 channels, 64x64 image
    output = block(x)
    print(f"ConvBlock Output Shape: {output.shape}")
    # Expecting 2 reductions of 2 pixels per convolution
    assert output.shape == (2, 64, 572, 572), "ConvBlock output shape mismatch."


def test_downscaling():
    """Test Downscaling for correct output dimensions."""
    down = Downscaling(in_channels=64, out_channels=128, dropout=0.1)
    x = torch.randn(2, 64, 572, 572)  # Batch of 2, 3 channels, 64x64 image
    output = down(x)
    print(f"Downscaling Output Shape: {output.shape}")
    # MaxPool reduces dimensions by 2, ConvBlock reduces by 4 (2 per conv layer)
    assert output.shape == (2, 128, 286, 286), "Downscaling output shape mismatch."


def test_start():
    """Test the start of the Unet"""
    block = ConvBlock(in_channels=3, out_channels=64, dropout=0.1)
    down = Downscaling(in_channels=64, out_channels=128, dropout=0.1)
    x = torch.randn(2, 3, 572, 572)  # Batch of 2, 3 channels, 64x64 image
    output = block(x)
    output = down(output)
    print(f"Downscaling Output Shape: {output.shape}")
    assert output.shape == (2, 128, 286, 286), "Downscaling output shape mismatch."


def test_upscaling():
    """Test Upscaling for correct output dimensions and skip connection."""
    up = Upscaling(in_channels=1024, out_channels=512, dropout=0.1)
    x = torch.randn(2, 1024, 32, 32)  # Upsampled feature map
    skip = torch.randn(2, 512, 64, 64)  # Skip connection
    output = up(x, skip)
    print(f"Upscaling Output Shape: {output.shape}")
    # Output should match skip connection dimensions after concatenation
    assert output.shape == (2, 512, 64, 64), "Upscaling output shape mismatch."


def test_odd_conv():
    """Test ConvBlock for correct output dimensions."""
    block = ConvBlock(in_channels=3, out_channels=64, dropout=0.1)
    down = Downscaling(in_channels=64, out_channels=128, dropout=0.1)
    down1 = Downscaling(in_channels=128, out_channels=256, dropout=0.1)
    down2 = Downscaling(in_channels=256, out_channels=512, dropout=0.1)
    x = torch.randn(2, 3, 572, 572)  # Batch of 2, 3 channels, 64x64 image
    output = block(x)
    output = down(output)
    output = down1(output)
    output = down2(output)
    print(f"ConvBlock Output Shape: {output.shape}")
    # Expecting 2 reductions of 2 pixels per convolution
    assert output.shape == (2, 512, 71, 71), "ConvBlock output shape mismatch."


def test_unet_forward_pass():
    """Test U-Net forward pass for correct output dimensions."""
    model = UNet(num_input_channels=3, num_output_classes=10, base_feature_size=64, dropout=0.1)
    x = torch.randn(2, 3, 256, 256)  # Batch of 2, 3 channels, 128x128 image
    output = model(x)
    print(f"UNet Forward Pass Output Shape: {output.shape}")
    # U-Net should return output with the same spatial dimensions as input
    assert output.shape == (2, 10, 256, 256), "U-Net output shape mismatch."
