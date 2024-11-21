import torch.nn as nn
import torch

class WrapperModel(nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model
        
    def forward(self, x):
        squeezed_tensor = torch.squeeze(x, 0)
        return self.model(squeezed_tensor)