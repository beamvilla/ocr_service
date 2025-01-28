import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from .eval import get_test_loss
from utils.log_utils import get_logger


def train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: nn.Module,
    loss_function: torch.nn.modules.loss,
    optimizer: torch.optim,
    device: torch.device,
    model_dir: str,
    epochs: int = 50,
    patience: int = 10
):
    def batch_train(
        images: DataLoader, 
        labels: DataLoader,
        model: nn.Module,
        loss_function,
        optimizer,
        device: torch.device
    ):
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)

        loss = loss_function(pred, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    min_loss = None
    not_improve = 0
    for epoch in range(epochs):
        for _, (images, labels) in enumerate(train_dataloader):
            batch_train(
                images=images,
                labels=labels,
                model=model,
                loss_function=loss_function,
                optimizer=optimizer,
                device=device
            )
        
       
        test_loss, acc = get_test_loss(
            val_dataloader=val_dataloader,
            model=model,
            loss_function=loss_function,
            device=device
        )
        get_logger().info(f"Epoch {epoch}, Test loss: {test_loss}, Acc: {acc}")

        if min_loss is None or test_loss < min_loss:
            min_loss = test_loss
            get_logger().info(f"Save model")
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
            not_improve = 0
        else:
            not_improve += 1

        if not_improve == patience:
            get_logger().info("Test loss not have improvement, then stop training.")