import torch
from torch import nn 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

N = 1000
X = np.random.random((N, 2)) * 6-3 
Y = np.cos(2*X[:,0]) + np.cos(3*X[:, 1])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:,1], Y)

plt.show()