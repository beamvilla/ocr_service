import torch
from torch import nn
import sys
sys.path.append("./")

import torch.utils
import torch.utils.data
from torch.utils.data import random_split, DataLoader

from src.models.text_recognition import (
    load_data, 
    TextRecognitionModel,
    train_model
)
from src.utils.log_utils import get_logger
from config.train_config import TrainConfig


train_config = TrainConfig("./config/train.yaml")

"""
Prepare dataset
"""
# Load data
get_logger().info("Load dataset.")
datasets = load_data(
    mnist_path="./dataset/mnist.npz",
    alpha_path="./dataset/alpha_data_60000.csv",
    input_size=(train_config.INPUT_WIDTH, train_config.INPUT_HEIGHT)
)

total_data = len(datasets)
train_size = int(0.8 * total_data)
test_size = total_data - train_size
train_dataset, test_dataset = random_split(datasets, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=train_config.BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=train_config.BATCH_SIZE, shuffle=True)

"""
Build model
"""
get_logger().info("Build model.")
text_recognition_model = TextRecognitionModel().to(train_config.DEVICE)

"""
Train model
"""
loss = nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = torch.optim.Adam(text_recognition_model.parameters(), lr=learning_rate)
get_logger().info("Training model.")
train_model(
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    model=text_recognition_model,
    loss_function=loss,
    optimizer=optimizer,
    device=train_config.DEVICE,
    model_dir=train_config.MODEL_DIR,
    epochs=train_config.EPOCHS,
    patience=train_config.PATIENCE
)
