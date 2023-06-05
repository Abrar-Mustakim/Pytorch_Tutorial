import numpy as np
import torch
from torch.autograd import Variable
from datetime import datetime

class batch_gd():
    def __init__(self, model, criterion, optimizer, train_loader, test_loader, epochs=5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def train(self):
        train_losses = np.zeros(self.epochs)
        test_losses = np.zeros(self.epochs)
        for i in range(self.epochs):
            t0 = datetime.now()
            train_loss = []
            for images, targets in self.train_loader:
                self.optimizer.zero_grad()
                images = Variable(images.view(-1, 28*28)).to(self.device)
                targets = Variable(targets).to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            train_loss = np.mean(train_loss)
            test_loss = []
            for images, targets in self.test_loader:
                #self.optimizer.zero_grad()
                images = Variable(images.view(-1, 28 * 28)).to(self.device)
                targets = Variable(targets).to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                test_loss.append(loss.item())
                #loss.backward()
                #self.optimizer.step()
            test_loss = np.mean(test_loss)
            train_losses[i] = train_loss
            test_losses[i] = test_loss
            dt = datetime.now() - t0
            print(f"Epochs {i + 1}/{self.epochs}, Train Loss: {train_loss:.4f}, \ Test Loss: {test_loss:.4f}, Duration: {dt}")
        return train_losses, test_losses