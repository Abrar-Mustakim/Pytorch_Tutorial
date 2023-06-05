import torch
from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dims, output_dims)
    def forward(self, X):
        return self.linear(X)

model = LogisticRegression(28*28, 10)
print(model)