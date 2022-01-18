import argparse
import sys
from omegaconf import OmegaConf
import numpy as np
import torch
from src.models.model import Network
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.data.make_dataset import MNISTDataset
import hydra
import logging
import wandb

@hydra.main(config_path="../configs", config_name="config.yaml")

def main(config):
    test_x = torch.randn([64,784])
    N_classes = 10
    
    model = Network(config["hyperparameters"]["input_size"],
                    config["hyperparameters"]["output_size"],
                    config["hyperparameters"]["hidden_layers"],
                    config["hyperparameters"]["dropout_probability"])
    
    out = model(test_x)

    assert out.shape[0] == test_x.shape[0]
    assert out.shape[1] == N_classes

if __name__ == '__main__':
    main()








