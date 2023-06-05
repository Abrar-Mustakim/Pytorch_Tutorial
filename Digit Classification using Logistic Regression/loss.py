from torch import nn
import torch


class CustomLoss(nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()


    def forward(self, outputs, targets):
        #targets = torch.LongTensor(targets)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        mask = float(10) #Class
        high_cost = (loss*mask).mean()
        return loss+high_cost

loss = CustomLoss()
#a = torch.Tensor([0, 0, 1, 0], requires_grad=True).type(torch.LongTensor).reshape(-1, 1)
a = torch.randn(3, 10, requires_grad=True)
b = torch.randn(3, 10)
m = nn.LogSoftmax(dim=1)
#print(a.shape)
print(loss(m(a), b))
print(a)
print(m(a))
print(b)