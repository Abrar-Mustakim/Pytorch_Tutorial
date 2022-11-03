import torch
from torch import nn 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

N = 1000
X = np.random.random((N, 2)) * 6-3 
Y = np.cos(2*X[:,0]) + np.cos(3*X[:, 1])


model = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

model.load_state_dict(torch.load("D:\PYTORCH for AI & ML\Artificial Neural Network in Pytorch\model2.pt"))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

with torch.no_grad():
    line = np.linspace(-5, 5 ,50)
    xx, yy = np.meshgrid(line, line)
    Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
    Xgrid_torch = torch.from_numpy(Xgrid.astype(np.float32))
    Yhat = model(Xgrid_torch).numpy().flatten()
    ax.plot_trisurf(Xgrid[:,0], Xgrid[:, 1], Yhat, linewidth=0.2, antialiased=True)
    plt.show()