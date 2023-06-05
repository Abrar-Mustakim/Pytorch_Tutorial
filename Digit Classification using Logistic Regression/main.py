import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import TrainDataset
from loss import CustomLoss
from model import LogisticRegression
from torch.utils.data import DataLoader
from train import batch_gd
from metrics import accuracy
from torch import nn
df = pd.read_csv("train.csv")
#print(df.head())

#image = df.iloc[:, 1:].values
#images = image[0].reshape(28, 28)

#plt.imshow(images)
#plt.show()

features = df.iloc[:, 1:].values
targets = df["label"].values

#print(features[0])
#print(targets[0])

#plt.imshow(features[10].reshape(28, 28))
#plt.title(targets[10])
#plt.show()
features = features / 255 #Normalization

features = features.astype(np.float32)
targets = targets.astype(np.float32)


#print(features[0])

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

train = TrainDataset(X_train, y_train)
test = TrainDataset(X_test, y_test)
#print(len(df["label"].unique()))

#DataLoader
batch_size = 100
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)



#Model
input_dims = 28*28
output_dims = 10
model = LogisticRegression(input_dims, output_dims)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#Optimizer
#loss = nn.CrossEntropyLoss()
loss = CustomLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training
train = batch_gd(model, loss, optimizer, train_loader, test_loader, epochs=50)
train_losses, test_losses = train.train()
weights =  "Digit Classifier model.pt"
torch.save(model.state_dict(), weights)


#Accuracy
Accuracy = accuracy(model, loss, optimizer, train_loader, test_loader, weights)
train_accuracy, test_accuracy = Accuracy.score()
print(f"Training Accuracy: {train_accuracy}, Testing Accuracy: {test_accuracy}")

#Plotting Loss Graphs
plt.plot(train_losses, "r", label="Training Loss")
plt.plot(test_losses, "b", label="Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Loss Function.png")
plt.show()



