import torch


def index_to_one_hot(tensor, num_classes=8):
    """
    Transforms a tensor with shape [N, W, H] containing index encoding
    into a one-hot encoded tensor with shape [N, num_classes, W, H].

    Args:
        tensor (torch.Tensor): Input tensor of shape [N, W, H] with index encoding.
        num_classes (int): Number of classes for one-hot encoding (default is 8).

    Returns:
        torch.Tensor: One-hot encoded tensor of shape [N, num_classes, W, H].
    """
    if tensor.dim() != 3:
        raise ValueError("Input tensor must have shape [N, W, H]")

    # Extract dimensions
    N, W, H = tensor.shape

    # Perform one-hot encoding
    one_hot = torch.zeros((N, num_classes, W, H), dtype=torch.long, device=tensor.device)
    one_hot.scatter_(1, tensor.unsqueeze(1), 1)

    return one_hot
