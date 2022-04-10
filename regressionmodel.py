from torch import nn
from torch.nn import functional as F
from tcn import define_model
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, tcn):
        super(NeuralNetwork, self).__init__()
        self.tcn = tcn
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    def forward(self, x):
        x = self.tcn(x)
        logits = self.linear_relu_stack(x)
        return logits

def create_model(tcndir,cuda):
    tcn = define_model(True)
    tcn.load_state_dict(torch.load(tcndir, map_location=lambda storage, loc: storage))

    #freeze the tcn model
    for param in tcn.parameters():
        param.requires_grad = False
    
    if cuda:
        return NeuralNetwork(tcn).cuda()
    else:
        return NeuralNetwork(tcn)
    