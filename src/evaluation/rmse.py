import torch 

def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2))