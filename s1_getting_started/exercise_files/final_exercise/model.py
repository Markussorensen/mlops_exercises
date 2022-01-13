from torch import nn


class MyAwesomeModel(nn.Module):
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
        
        
    def forward(self, x):
        x = self.layers(x.float())
        return x