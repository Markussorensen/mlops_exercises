import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.make_dataset import MNISTDataset
import hydra


@hydra.main(config_path="../configs", config_name="config.yaml")

def main(config):
    N_train = 25000
    N_test = 5000


    train_data = torch.load(config["folders"]["project"] + "data/processed/train.pt")
    test_data = torch.load(config["folders"]["project"] + "data/processed/test.pt")
    trainloader = DataLoader(train_data, batch_size=config["hyperparameters"]["batch_size"], shuffle=True)
    testloader = DataLoader(test_data, batch_size=config["hyperparameters"]["batch_size"], shuffle=True)

    assert len(train_data) == N_train and len(test_data) == N_test
    for images, labels in trainloader:
        assert images.shape[1] == 784
    for images, labels in testloader:
        assert images.shape[1] == 784
    represented_train_labels = set(train_data[:][1].numpy())
    represented_test_labels = set(test_data[:][1].numpy())
    for i in range(10):
        assert i in represented_train_labels
        assert i in represented_test_labels


if __name__ == '__main__':
    main()
