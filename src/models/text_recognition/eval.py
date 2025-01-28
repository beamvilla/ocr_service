import torch
from torch.utils.data import DataLoader
from torch import nn
from typing import Tuple


def get_test_loss(
    val_dataloader: DataLoader, 
    model: nn.Module, 
    loss_function: torch.nn.modules.loss,
    device: torch.device
) -> Tuple[float, float]:
    num_batches = len(val_dataloader)
    total = len(val_dataloader.dataset)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)
            test_loss += loss_function(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= total
    return test_loss, correct