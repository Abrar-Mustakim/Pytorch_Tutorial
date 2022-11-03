import torch
from torch import nn
import matplotlib.pyplot as plt 
import numpy as np 
import torchvision 
import torchvision.transforms as transforms


tran_dataset = torchvision.datasets.MNIST(
    root = "MNIST",
    train=True,
    transform= transforms.ToTensor(),
    download=False
)
