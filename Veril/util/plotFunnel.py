import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from util import samples

def plotFunnel(x, V):
  xlim = np.load('../data/VDP/VanDerPol_limitCycle.npy')
  fig, ax = plt.subplots()
  bdry = ax.plot(xlim[0, :], xlim[1, :], label='ROA boundary')
  x = samples.levelsetData(x,V)[0]
  plt.fill(x[:,0], x[:,1])
  plt.show()

# def plotNew(x,V):
#   xlim = np.load('../data/VanDerPol_limitCycle.npy')
#   fig, ax = plt.subplots()
#   bdry = ax.plot(xlim[0, :], xlim[1, :], label='ROA boundary')
#   samples = get_data(d=3, num_grid=100)[0]
#   points = np.zeros((1, 2))
#   for s in samples:
#       env = dict(zip(x, s.T))
#       if V.Evaluate(env) < 1:
#           points = np.vstack((points, s))
#   plt.scatter(points[:,0],points[:,1])
#   plt.show()
