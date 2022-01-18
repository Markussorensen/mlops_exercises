import argparse
import sys
from omegaconf import OmegaConf
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from model import Network
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.data.make_dataset import MNISTDataset
import hydra
import logging
import wandb

wandb.login()
@hydra.main(config_path="../../configs", config_name="config.yaml")

def main(config):
    with wandb.init(project="linear_model", config=dict(config["hyperparameters"])):
        torch.seed(config["hyperparameters"]["seed"])
        train_data = torch.load(config["folders"]["project"] + "data/processed/train.pt")
        test_data = torch.load(config["folders"]["project"] + "data/processed/test.pt")
        trainloader = DataLoader(train_data, batch_size=wandb.config["batch_size"], shuffle=True)
        testloader = DataLoader(test_data, batch_size=wandb.config["batch_size"], shuffle=True)

        model = Network(wandb.config["input_size"],
                        wandb.config["output_size"],
                        wandb.config["hidden_layers"],
                        wandb.config["dropout_probability"])
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])
        epochs = wandb.config["epochs"]

        wandb.watch(model, criterion, log="all", log_freq=10)

        train_losses = []
        test_losses = []
        for e in range(epochs):
            running_loss = 0
            ### Training the model
            model.train()
            for images, labels in trainloader:

                optimizer.zero_grad()
                
                log_ps = model(images.float())
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            train_losses.append(running_loss/len(trainloader.dataset))
            
            ### Validating the model
            model.eval()
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images, labels in testloader:            
                    log_ps = model(images.float())
                    loss = criterion(log_ps, labels)
                    test_loss += loss.item()
                    ps = torch.exp(log_ps)
                    # Class with highest probability is our predicted class, compare with true label
                    equality = (labels.data == ps.max(1)[1])
                    # Accuracy is number of correct predictions divided by all predictions, just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
            test_losses.append(test_loss/len(testloader.dataset))
            
            log.info(f"Epoch: {e+1} of {epochs}, Training Loss: {running_loss/len(trainloader.dataset)}, Test Loss: {test_loss/len(testloader.dataset)}, Test Accuracy: {(accuracy/len(testloader))}")
            wandb.log({"epoch":e+1, "training_loss": running_loss/len(trainloader.dataset), "test_loss": test_loss/len(testloader.dataset), "test_accuracy": accuracy/len(testloader), "last_img": wandb.Image(images[0].view(-1,config["data"]["img_size"],config["data"]["img_size"])), "last_outputs": wandb.Histogram(log_ps)}, step=e)
        torch.save(model.state_dict(), config["folders"]["project"] + "models/linear/linear_weights.pt")
        torch.onnx.export(model, images, config["folders"]["project"] + "models/linear/linear_model.onnx")
        wandb.save("linear_model.onnx")

    plt.figure()
    plt.plot(range(epochs),train_losses, label="train loss")
    plt.plot(range(epochs),test_losses, label="test loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(config["folders"]["project"] + "reports/figures/linear_model.pdf")

if __name__ == '__main__':
    log = logging.getLogger(__name__)
    main()








