import torch
import torch.nn.functional as F
from torch import nn, optim
from pytorch_lightning import LightningModule


# class MyAwesomeModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Inputs to hidden layer linear transformation
#         self.layers = nn.Sequential(
#                                     nn.Linear(784,256), nn.ReLU(), nn.Dropout(0.2),
#                                     nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2),
#                                     nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.2),
#                                     nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.2),
#                                     nn.Linear(32,10), nn.Softmax(dim=1),                                                                        
#         )
        
        
#     def forward(self, x):
#         x = self.layers(x.float())
#         return x

class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.layers = nn.Sequential(
                                    nn.Linear(784,256), nn.ReLU(), nn.Dropout(0.2),
                                    nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2),
                                    nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.2),
                                    nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.2),
                                    nn.Linear(32,10), nn.Softmax(dim=1),                                                                        
        )

        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.layers(x.float())
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        out = self(data)
        loss = self.criterion(out, target)
        return loss
    
    def configure_optimizers(self, lr=0.003):
        return optim.Adam(self.parameters(),lr=lr)

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)