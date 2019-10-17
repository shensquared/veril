import plotly.graph_objects as go


def call_CLsys(sys, tm1, num_samples):
    plant = Plants.get(sys.plant_name, sys.dt, sys.obs_idx)
    inputs = K.placeholder()
    states = [K.placeholder(shape=(num_samples, plant.num_states)), K.placeholder
              (shape=(num_samples, sys.units))]
    [x_tm2, c_tm2] = sys.cell.call(inputs, states, training=False)[1]
    feed_dict = dict(zip(states, tm1))
    sess = K.get_session()
    x_tm2 = sess.run(x_tm2, feed_dict=feed_dict)
    c_tm2 = sess.run(c_tm2, feed_dict=feed_dict)
    return [x_tm2, c_tm2]


def simulate(CL, timesteps, init, num_samples):
    for i in range(timesteps):
        init = call_CLsys(CL, init, num_samples)
    return init


def do_plotting(CL, sim=True):
    num_samples = 10000
    u = np.linspace(-4, 4, np.sqrt(num_samples))
    v = np.linspace(-2, 2, np.sqrt(num_samples))
    u, v = np.meshgrid(u, v)
    u, v = u.flatten(), v.flatten()
    init_x_train = np.array([u.flatten(), v.flatten()]).T
    init_c = np.zeros((num_samples, num_units))
    if sim:
        final = simulate(CL, 100, [init_x_train, init_c], num_samples)
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
