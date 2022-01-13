import argparse
import sys

import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms

from data import mnist
from model import MyAwesomeModel

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.003)
        parser.add_argument('--epochs', default=200)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        model.train()
        train_set, _ = mnist()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        epochs = args.epochs

        train_losses = []
        for e in range(epochs):
            running_loss = 0
            for images, labels in train_set:
                if running_loss == 0:
                    ps = torch.exp(model(images))
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy = torch.mean(equals.type(torch.FloatTensor))
                    print(f'Accuracy: {accuracy.item()*100}%, Epoch: {e}')
                
                optimizer.zero_grad()
                
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            train_losses.append(running_loss)

        torch.save(model.state_dict(), 'C:/Users/Marku/OneDrive - Danmarks Tekniske Universitet/Studie/7. Semester/Machine Learning Operations/dtu_mlops/s1_getting_started/exercise_files/final_exercise/weights/trained_model.pt')
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model = torch.load(args.load_model_from)
        model.eval()
        _, test_set = mnist()
        criterion = nn.CrossEntropyLoss()

        validation_loss = 0
        for images, labels in test_set:
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            
            validation_loss += loss.item()

        return validation_loss

if __name__ == '__main__':
    TrainOREvaluate()
    

    
    
    
    
    
    
    
    
    