import plotly.graph_objects as go


def do_plotting(CL, sim=True):
    if sim:
        final = simOneTraj(CL, 100, [init_x_train, init_c], num_samples)
        finalx = final[0]
        # TODO: extract plant name from the CL
        np.save('../data/double_sim_100steps.npy', finalx)
    else:
        finalx = np.load('../data/double_sim.npy')

    # final_theta = np.arctan2(finalx[:,0],finalx[:,1])
    z = finalx[:, 1]**2 + finalx[:, 0]**2
    fig = go.Figure(data=[go.Scatter3d(x=u, y=v, z=z,
                                       mode='markers', marker=dict(
                                           size=2,
                                           color=z,                # set color to an array/list of desired values
                                           colorscale='Viridis',   # choose a colorscale
                                           opacity=0.8
                                       ))])
    fig.update_layout(scene=dict(
        xaxis_title='theta',
        yaxis_title='thetadot',
        zaxis_title='final deviation'),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10))
    fig.show()
