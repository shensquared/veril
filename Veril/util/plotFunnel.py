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