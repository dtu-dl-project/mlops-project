from segmentationsuim.model import Downscaling
import torch

def test_downscaling():
    block = Downscaling(3, 64)

    x = torch.zeros(1, 3, 256, 256)
    y = block(x)

    assert y.shape == (1, 64, 128, 128)
