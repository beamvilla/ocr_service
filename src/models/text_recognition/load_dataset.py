import numpy as np
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_mnist_data(dataset_path: str) -> Tuple[np.array]:
    mnist_dataset = np.load(dataset_path, allow_pickle=True)
    x_train = mnist_dataset["x_train"]
    y_train = mnist_dataset["y_train"]

    x_test = mnist_dataset["x_test"]
    y_test = mnist_dataset["y_test"]
    return x_train, y_train, x_test, y_test


def load_data(
    mnist_path: str = "./dataset/mnist.npz",
    alpha_path: str = "./dataset/A_Z Handwritten Data.csv"
) -> TensorDataset:
    # MNIST Data
    digit_image_train, digit_label_train, digit_image_test, digit_label_test = load_mnist_data(mnist_path)
    digit_label = np.hstack((
        digit_label_train, 
        digit_label_test
    ))
    digit_image = np.vstack((
        digit_image_train,
        digit_image_test
    ))
   
    #A-Z Data
    alpha_data = pd.read_csv(alpha_path)
    alpha_label = np.array(alpha_data["0"])  # The '0' column is the target
    alpha_label += 10  # A-Z will not overlab with MNIST
    alpha_image = alpha_data.drop(["0"], axis=1)  # Drop target column
    alpha_image = np.reshape(
        alpha_image.values,
        (alpha_image.shape[0], 28, 28)
    ) # Resize to (28*28)

    # Combine datasets
    labels = np.hstack((digit_label, alpha_label))
    images = np.vstack((digit_image, alpha_image))

    images = images.astype(np.float32) / 255.0
    images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # Ex shape: (total, 1, 28, 28)
    # Convert labels to tensors
    labels = torch.tensor(labels, dtype=torch.long)  # Ex shape: (total,)
    # Create a Dataset
    dataset = TensorDataset(images, labels)
    return dataset


# def get_dataloader(
#     images: np.array, 
#     labels: np.array,
#     batch_size: int = 32
# ) -> DataLoader:
#     images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # Ex shape: (total, 1, 28, 28)

#     # Convert labels to tensors
#     labels = torch.tensor(labels, dtype=torch.long)  # Ex shape: (total,)

#     # Create a Dataset and DataLoader
#     dataset = TensorDataset(images, labels)

#     # Use DataLoader for batching and shuffling
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     return dataloader