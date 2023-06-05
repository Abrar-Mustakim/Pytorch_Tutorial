import torch
from model import LogisticRegression
from torch.autograd import Variable
import numpy as np


class accuracy():
    def __init__(self, model, criterion, optimizer, train_loader, test_loader, weights):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.weights = weights
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def model(self):

        self.input_dims = 28 * 28
        self.output_dims = 10
        self.model = LogisticRegression(self.input_dims, self.output_dims)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.model.load_state_dict(torch.load(self.weights))

        return self.model


    def score(self):



        correct = 0
        n_total = 0
        for images, targets in self.train_loader:
            self.optimizer.zero_grad()
            images = Variable(images.view(-1, 28*28)).to(self.device)
            targets = Variable(targets).to(self.device)
            outputs = self.model(images)

            _, predictions = torch.max(outputs, 1)

            correct += (targets == predictions).sum().item()
            n_total += targets.shape[0]

        train_accuracy = correct / n_total

        correct = 0
        n_total = 0

        for images, targets in self.test_loader:
            #self.optimizer.zero_grad()
            images = Variable(images.view(-1, 28 * 28)).to(self.device)
            targets = Variable(targets).to(self.device)
            outputs = self.model(images)
            _, predictions = torch.max(outputs, 1)

            correct += (targets == predictions).sum().item()
            n_total += targets.shape[0]


            #loss.backward()
            #self.optimizer.step()
        test_accuracy = correct / n_total

        return train_accuracy, test_accuracy


