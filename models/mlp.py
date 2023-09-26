import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, num_layers, layer_size):

        super().__init__()

        layers = []
        for i in range(num_layers + 1):
            if i == 0:
                layers.append(nn.Linear(in_dim, layer_size))
                layers.append(nn.ReLU())
            elif i == num_layers:
                layers.append(nn.Linear(layer_size, out_dim))
            else:
                layers.append(nn.Linear(layer_size, layer_size))
                layers.append(nn.ReLU())

        self.fc = nn.Sequential(*layers)

    def forward(self, x):

        return self.fc(x)