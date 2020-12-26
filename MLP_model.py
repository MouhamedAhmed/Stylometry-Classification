import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=2),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)