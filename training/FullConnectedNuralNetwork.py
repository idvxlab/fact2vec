import torch.nn as nn


class FullConnectedNuralNetwork(nn.Module):
    def __init__(self):
        super(FullConnectedNuralNetwork, self).__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(768, 768, bias=True),
            nn.Linear(768, 768, bias=True),
        )

    def forward(self, x):
        fc1 = self.hidden1(x)
        return fc1
