import torch
from torch.utils.data import DataLoader
from torch import nn


def get_test_loss(
    val_dataloader: DataLoader, 
    model: nn.Module, 
    loss_function: torch.nn.modules.loss,
    device: torch.device
) -> float:
    num_batches = len(val_dataloader)
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)
            test_loss = loss_function(pred, labels).item()

    test_loss /= num_batches
    return test_loss