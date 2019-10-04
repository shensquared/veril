import plotly.graph_objects as go
import numpy as np

def do_plot(x,y,z)
u = np.linspace(-np.pi, np.pi, np.sqrt(num_samples))
v = np.linspace(-1, 1, np.sqrt(num_samples))
u, v = np.meshgrid(u, v)
theta, thetadot = u.flatten(), v.flatten()
init_x_train = np.array([np.sin(theta), np.cos(theta), thetadot]).T


fig = go.Figure(data=[go.Scatter3d(x=theta, y=thetadot, z=z,
                                   mode='markers')])
fig.show()
