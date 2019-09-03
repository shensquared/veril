import numpy as np
# from control.matlab import drss
# from scipy import integrate
# import matplotlib.pyplot as plt
# import os
from keras import backend as K
import tensorflow as tf
import six


def get(plant_name, dt, obs_idx):
    if isinstance(plant_name, six.string_types):
        identifier = str(plant_name)
        return globals()[identifier](dt=dt, obs_idx=obs_idx)

class Plant():

    def __init__(self, dt=1e-3, obs_idx=None, num_disturb=0):
        self.dt = dt
        self.num_disturb = num_disturb

    def step(self, x, u):
        pass

    def get_states(self):
        return self.states

    def get_obs(self, x):
        if self.obs_idx is None:
            return x
        else:
            return tf.gather(x, self.obs_idx, axis=1)

    def np_get_obs(self, x):
        if self.obs_idx is None:
            return x
        else:
            return x[self.obs_idx]


class Pendulum(Plant):

    def __init__(self, dt=1e-3, obs_idx=None, num_disturb=0):
        self.name = 'Pendulum'
        self.m = 1
        self.l = .5
        self.b = 0.1
        self.lc = .5
        self.I = .25
        self.g = 9.81

        # b/c we deal with the [sin cos thetadot] coordinate
        self.num_states = 3
        self.num_inputs = 1
        self.obs_idx = obs_idx
        if obs_idx is None:
            self.num_outputs = self.num_states
        else:
            self.num_outputs = len(obs_idx)
        self.num_disturb = num_disturb

        self.dt = dt
        self.x0 = np.array([0, -1, 0])
        self.y0 = self.np_get_obs(self.x0)
        self.u0 = 0

    def step(self, x, u):
        s = tf.gather(x, [0], axis=1)
        c = tf.gather(x, [1], axis=1)
        thetadot = tf.gather(x, [2], axis=1)

        # desired fixed point should be sin(pi)=0, cos(pi)=-1, thetadot =0
        delta = K.concatenate([c * thetadot,
                               -s * thetadot,
                               (-self.b * thetadot + u) / (self.m * self.l * self.l) -
                               self.g * s / self.l])
        self.states = x + delta * self.dt
        return self.states

    def np_step(self, x, u):
        [s, c, thetadot] = x
        # desired fixed point should be sin(pi)=0, cos(pi)=-1, thetadot =0

        delta = np.array([c * thetadot,
                          -s * thetadot,
                          (-self.b * thetadot + u[0]) / (self.m * self.l * self.l) -
                          self.g * s / self.l])
        self.states = x + delta * self.dt
        return self.states

    def get_data(self, num_samples, timesteps, num_units):
        u = np.linspace(-np.pi, np.pi, np.sqrt(num_samples))
        v = np.linspace(-1, 1, np.sqrt(num_samples))
        u, v = np.meshgrid(u, v)
        theta, thetadot = u.flatten(), v.flatten()

        # init_theta = np.random.uniform(np.pi-.1,np.pi+.1, (num_samples, 1))
        # init_thetadot = np.random.uniform(-1, 1, (num_samples, 1))

        init_x_train = np.array([np.sin(theta), np.cos(theta), thetadot]).T
        # init_c_train = np.random.uniform(-1, 1, (num_samples, num_units))
        init_c_train = np.zeros((num_samples, num_units))
        ext_in_train = np.zeros((num_samples, timesteps, self.num_disturb))
        x_train = [init_x_train, init_c_train, ext_in_train]

        # y_train = np.zeros((num_samples, self.num_states))
        # cos(pi)=-1
        # y_train[:, 1] = -np.ones((num_samples,))
        y_train = np.tile(self.y0, (num_samples, 1))
        return x_train, y_train


class Satellite(Plant):

    def __init__(self, dt=1e-3, obs_idx=None, num_disturb=0):
        self.name = 'Satellite'
        self.num_states = 6
        self.num_inputs = 3
        self.obs_idx = obs_idx
        if obs_idx is None:
            self.num_outputs = self.num_states
        else:
            self.num_outputs = len(obs_idx)

        self.dt = dt
        self.num_disturb = num_disturb
        self.x0 = np.array([0, 0, 0, 0, 0, 0])
        self.y0 = self.np_get_obs(self.x0)
        self.u0 = 0

    # def reset(self, lb=-2.5, ub=2.5):
        # np.random.seed(3)
        # self.states = np.random.uniform(lb, ub, (1,self.num_states))
        # self.states = K.random_uniform((self.num_states,), lb, ub)

    def Sigma(self, alpha):
        # pass
        # accepts alpha of shape (None,3)
        # [a1, a2, a3] = alpha
        # return np.array([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])
        sigma = K.variable([[0, -alpha[0, 2], alpha[0, 1]], [alpha[0, 2], 0, -alpha
                                                             [0, 0]], [-alpha[0, 1], alpha[0, 0], 0]])
        return sigma

    def step(self, x, u):
        H = K.constant(np.diag([2, 1, .5]))
        H_inv = K.constant(np.diag([.5, 1, 2]))
        eye3 = K.constant(np.eye(3))

        w = x[:, :3]
        phi = x[:, 3:6]

        # CT dynamics:
        # H*dot(w)=-Sigma(w)*H*w+u, or equivalently:
        # dot(w)=H_inv*(-Sigma(w)*H*w+u), forward Euler
        # w2=w+(H_inv.dot(np.dot(-self.Sigma(w),np.dot(H,w))+u))*self.dt
        _ = K.transpose(K.dot(H, K.transpose(w)))
        _ = self.dt * K.dot(H_inv, -K.transpose(tf.cross(w, _)))
        _ = K.transpose(_)
        w2 = (w + _) + K.transpose(K.dot(H_inv, K.transpose(u)) * self.dt)

        # dot(phi)=.5*(eye+phi*phi.T+Sigma(phi))*w
        _ = K.dot(K.transpose(phi), phi)
        _ = .5 * K.dot(_ + eye3, K.transpose(w)) * self.dt
        _ = _ + .5 * K.transpose(tf.cross(phi, w)) * self.dt
        phi2 = phi + K.transpose(_)

        # for comparing dot and cross:
        # self._compare_dotcross(x,u)
        # print((K.eval(K.concatenate([w2,phi2]))))

        # for comparing keras and numpy behaviors:
        # np_u = np.array([4,8,7])
        # np_u = np.reshape(np_u,(3,1))
        # np_w2=w+(H_inv.dot(np.dot(-self.Sigma(w),np.dot(H,w))+np_u))*self.dt
        # tensor_w2=K.eval(w2)

        self.states = K.concatenate([w2, phi2])
        return self.states

    def _get_openu(self):
        return self.open_u

    def get_data(self, num_samples, timesteps, num_units, lb=-1, ub=1):
        # ticks= np.power(num_samples,self.num_states)
        # x1 = np.linspace(lb,ub, ticks)
        # x2 = np.linspace(lb,ub, ticks)
        # x3 = np.linspace(lb,ub, ticks)
        # x4 = np.linspace(lb,ub, ticks)
        # x5 = np.linspace(lb,ub, ticks)
        # x6 = np.linspace(lb,ub, ticks)

        # x1,x2,x3,x4,x5,x6 = np.meshgrid(x1,x2,x3,x4,x5,x6)
        # x1,x2,x3,x4,x5,x6 = x1.flatten(),x2.flatten(),x3.flatten(),x4.flatten(),x5.flatten(),x6.flatten()
        # init_x_train = np.array([x1,x2,x3,x4,x5,x6]).T

        init_x_train = np.random.uniform(
            lb, ub, (num_samples, self.num_states))
        init_c_train = np.zeros((num_samples, num_units))
        ext_in_train = np.zeros((num_samples, timesteps, self.num_disturb))
        x_train = [init_x_train, init_c_train, ext_in_train]
        y_train = np.tile(self.x0, (num_samples, 1))
        return x_train, y_train

    def _compare_dotcross(self, x, u):
        H = K.constant(np.diag([2, 1, .5]))
        H_inv = K.constant(np.diag([.5, 1, 2]))
        eye3 = K.constant(np.eye(3))
        w = x[:, :3]
        phi = x[:, 3:6]
        # CT dynamics:
        # H*dot(w)=-Sigma(w)*H*w+u, or equivalently:
        # dot(w)=H_inv*(-Sigma(w)*H*w+u), forward Euler
        # w2=w+(H_inv.dot(np.dot(-self.Sigma(w),np.dot(H,w))+u))*self.dt
        # _ = np.linalg.multi_dot([H_inv,-self.Sigma(w),H,w])*self.dt
        # dot prod realization
        _ = K.transpose(K.dot(K.dot(K.dot(H_inv, -self.Sigma(w)), H), K.transpose
                              (w))) * self.dt
        w2 = (w + _) + K.transpose(K.dot(H_inv, K.transpose(u)) * self.dt)

        # dot(phi)=.5*(eye+phi*phi.T+Sigma(phi))*w
        # dot product realization
        phi2 = phi + K.transpose(.5 * K.dot((eye3 + self.Sigma(phi) + K.dot(K.transpose
                                                                            (phi), phi)), K.transpose(w))) * self.dt
        print(K.eval(K.concatenate([w2, phi2])))


class VanderPol():

    def __init__(self, dt, num_states=2, num_inputs=1, num_outputs=1):
        self.name = 'VDP'
        self.num_states = 2
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.dt = dt
        self.open_u = np.zeros((num_inputs,))

    def reset(self, stable_sample):
        in_true_ROA = False
        if stable_sample:
            while not in_true_ROA:
                x = np.random.uniform(-2.5, 2.5, (self.num_states,))
                in_true_ROA = self.in_true_ROA(np.array([x]))[0]
            self.states = x
        else:
            self.states = np.random.uniform(-2.5, 2.5, (self.num_states,))

    def in_true_ROA(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        V = (1.8027e-06) + (0.28557) * x1**2 + (0.0085754) * x1**4 + (0.18442) * x2**2 + (0.016538) * x2**4 + \
            (-0.34562) * x2 * x1 + (0.064721) * x2 * x1**3 + \
            (0.10556) * x2**2 * x1**2 + (-0.060367) * x2**3 * x1
        rho = 1.1
        in_true_ROA = (V < rho)
        return in_true_ROA

    # def step_once(self, full_states):
    #     x1, x2 = self.states
    #     dt = self.dt
    #     x1_next = x1 - x2 * dt
    #     x2_next = x2 + (x1 + x2 * x1 * x1 - x2) * dt
    #     self.states = np.array([x1_next, x2_next])
    #     return self._get_obs(full_states)

    # def sim_traj(self, timesteps, full_states, stable_sample):
    #     self.reset(stable_sample)
    # return np.array([self.step_once(full_states) for i in range(timesteps)])

    # [            -x2  ]
    # [  x1-x2+x2*x1**2  ]

    def inward_vdp(self, t, y):
        return - np.array([y[1], (1 - y[0]**2) * y[1] - y[0]])

    def outward_vdp(self, t, y):
        return - self.inward_vdp(t, y)

    def step_once(self, full_states):
        sol = integrate.RK45(self.inward_vdp, 0, self.states, self.dt)
        self.states = sol.y
        return self._get_obs(full_states)

    def sim_traj(self, timesteps, full_states, stable_sample,
                 given_initial=None, scale_time=1):
        if given_initial is None:
            self.reset(stable_sample)
        else:
            self.states = given_initial
        # sol = integrate.RK45 (self.inward_vdp,0,self.initial_states, self.dt)
        # if full_states:
        #     return sol
        # else:
        #     return sol[:, -1].reshape(timesteps, 1)
        # return np.array([self.step_once(full_states) for i in
        # range(timesteps)])
        sol = integrate.solve_ivp(self.inward_vdp, [0, scale_time * self.dt *
                                                    timesteps], self.states, t_eval=np.linspace(0, self.dt * timesteps *
                                                                                                scale_time, timesteps), rtol=1e-9)
        # atol=np.max(np.abs(self.states))
        if sol.status is not 0:
            print(self.states)
        # asserting the integration reaches the final time stamps
        assert sol.status <= 0

        # sol.y = np.flip(sol.y,axis=1)
        if full_states:
            return sol.y.T
        else:
            # return np.array([sol.y.T[:, -1]]).reshape(timesteps, 1)
            return np.array([sol.y.T[:, -1]]).reshape(sol.y.T.shape[0], 1)

    def _phase_portrait(self, ax, ax_max):
        num = 60
        u = np.linspace(-ax_max, ax_max, num=num)
        v = np.linspace(-ax_max, ax_max, num=num)
        u, v = np.meshgrid(u, v)
        u, v = u.flatten(), v.flatten()
        x = -v
        y = u - v + v * u**2
        # angles='xy', width=1e-3,scale_units='xy', scale=12, color='r'
        ax.quiver(u, v, x, y, color='r', width=1e-3, scale_units='x')

    def limit_cycle(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        dirname = os.path.dirname(__file__)
        dirname = os.path.join(dirname, '../data/')
        file_name = dirname + 'VanDerPol_limitCycle.npy'
        xlim = np.load(file_name).T
        # self._phase_portrait(ax, 3)
        # limit cycle starting point (copied from drake, no record of origin or
        # derivation) [-0.1144,2.0578]
        # sol = integrate.solve_ivp(self.inward_vdp, [0, self.dt * timesteps],
        #                           [-0.1145, 2.05], t_eval=np.arange(0, self.dt * timesteps, self.dt))
        # ax.scatter(sol.y.T[:, 0], sol.y.T[:, 1], c='b')
        # ax.scatter(sol.y.T[0, 0], sol.y.T[0, 1], c='r')
        ax.plot(xlim[:, 0], xlim[:, 1], c='#A31F34', label='True ROA boundary')
        if ax is None:
            plt.show()

    def plot_sim_traj(self, timesteps, full_states, stable_sample,
                      num_samples=1, scale_time=1):
        fig, ax = plt.subplots()
        for i in range(num_samples):
            sim_traj = self.sim_traj(timesteps, full_states,
                                     stable_sample, scale_time=scale_time)
            if full_states:
                self._phase_portrait(ax, 3)
                ax.scatter(sim_traj[:, 0], sim_traj[:, 1], c='b')
                ax.scatter(sim_traj[0, 0], sim_traj[0, 1], c='g')
                # ax.axis([-3, 3, -3, 3])
                plt.xlabel('$x_1$')
                plt.ylabel('$x_2$', rotation=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
            else:
                t = np.arange(timesteps)
                ax.scatter(t, sim_traj, c='b')
                plt.xlabel('$t$')
                plt.ylabel('$x_2$', rotation=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
        plt.show()

    def _get_obs(self, full_states):
        if full_states:
            return self.states
        else:
            # return x2
            return [self.states[-1]]

    def get_openu(self):
        return self.open_u

    def explicit_Jacobian(self):
        x1 = Variable('x1')
        x2 = Variable('x2')
        x = np.vstack((x1, x2))
        dt = self.dt
        x1_next = x1 - x2 * dt
        x2_next = x2 + (x1 + x2 * x1 * x1 - x2) * dt

        xplus = np.vstack((x1_next, x2_next))
        Jac = Jacobian(xplus, x)
        # evaluate at zero
        env = {_: 0 for _ in np.hstack((x1, x2))}
        # A0: the system linearized around the origin
        A0 = np.array(
            [[x.Evaluate(env) for x in J] for J in Jac])

        if (np.abs(eig(A0)[0]) <= 1).all():
            print('linearized stable')
            print(np.abs(eig(A0)[0]))
            self.A0 = A0
        else:
            print('linerized not stable')
            print(np.abs(eig(A0)[0]))
            self.A0 = .5 * np.eye(2)
        # scipy solves for A'XA-X=-Q
        self.S0 = solve_discrete_lyapunov(self.A0, np.eye(2), method='direct')


class DoubleIntegrator():

    def __init__(self, num_states=2, num_inputs=1, num_outputs=1):
        self.name = 'DoubleIntegrator'
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def step_once(self, full_states, u):
        x1, x2 = self.states
        x1_next = x1 + x2
        x2_next = x2 + u
        self.states = np.array([x1_next, x2_next])
        return self._get_obs(full_states)

    def sim_traj(self, timesteps, full_states, stable_sample, u=0,
                 scale_time=1, given_initial=None):
        if given_initial is None:
            self.reset()
        else:
            self.states = given_initial
        return np.array([self.step_once(full_states, u) for i in range
                         (timesteps)])

    def _get_obs(self, full_states):
        if full_states:
            return self.states
        else:
            # return last states
            return [self.states[0]]

    def reset(self):
        self.states = np.random.uniform(-5, 5, (self.num_states,))

    def plot_sim_traj(self, timesteps, full_states, stable_sample,
                      num_samples=1, scale_time=1):
        fig, ax = plt.subplots()
        for i in range(num_samples):
            sim_traj = self.sim_traj(timesteps, full_states,
                                     stable_sample, scale_time=scale_time)
            if full_states:
                ax.scatter(sim_traj[:, 0], sim_traj[:, 1], c='b')
                ax.scatter(sim_traj[0, 0], sim_traj[0, 1], c='g')
                # ax.axis([-3, 3, -3, 3])
                plt.xlabel('$x_1$')
                plt.ylabel('$x_2$', rotation=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
            else:
                t = np.arange(timesteps)
                ax.plot(t, sim_traj, c='b')
                plt.xlabel('$t$')
                plt.ylabel('$x_2$', rotation=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
        plt.show()


class HybridPlant():

    def __init__(self, num_states=2, num_inputs=1, num_outputs=1):
        self.name = 'HybridPlant'
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def step_once(self, full_states):
        x1, x2 = self.states
        if x2 <= 0:
            x1_next = .5 * x1 + x1 * x2
            x2_next = -0.8 * x2
        else:
            x1_next = 0.5 * x1
            x2_next = -0.8 * x2 - x1**2
        self.states = np.array([x1_next, x2_next])
        return self._get_obs(full_states)

    def sim_traj(self, timesteps, full_states, stable_sample,
                 scale_time=1, given_initial=None):
        if given_initial is None:
            self.reset()
        else:
            self.states = given_initial
        return np.array([self.step_once(full_states) for i in range(timesteps)])

    def _get_obs(self, full_states):
        if full_states:
            return self.states
        else:
            # return last states
            return [self.states[-1]]

    def reset(self):
        self.states = np.random.uniform(-1, 1, (self.num_states,))

    def plot_sim_traj(self, timesteps, full_states, stable_sample,
                      num_samples=1, scale_time=1):
        fig, ax = plt.subplots()
        for i in range(num_samples):
            sim_traj = self.sim_traj(timesteps, full_states,
                                     stable_sample, scale_time=scale_time)
            if full_states:
                ax.scatter(sim_traj[:, 0], sim_traj[:, 1], c='b')
                ax.scatter(sim_traj[0, 0], sim_traj[0, 1], c='g')
                # ax.axis([-3, 3, -3, 3])
                plt.xlabel('$x_1$')
                plt.ylabel('$x_2$', rotation=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
            else:
                t = np.arange(timesteps)
                ax.plot(t, sim_traj, c='b')
                plt.xlabel('$t$')
                plt.ylabel('$x_2$', rotation=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
        plt.show()

    def plot_sampled_ROA(self, full_states):
        fig, ax = plt.subplots()
        num = 100
        u = np.linspace(-3, 3, num=num)
        v = np.linspace(-3, 3, num=num)
        u, v = np.meshgrid(u, v)
        u, v = u.flatten(), v.flatten()
        x = np.array([u, v]).T

        for i in range(x.shape[0]):
            traj = []
            self.states = x[i]
            for j in range(20000):
                if (np.abs(self.states) <= 1e3).all():
                    next_obs = self.step_once(full_states)
                    traj.append(next_obs)
                else:
                    x[i] = np.NaN
                    traj = None
                    break
            if traj is not None:
                traj = np.array(traj)
                if (np.abs(traj[:, -1]) >= 1e-1).any():
                    x[i] = np.NaN

        ax.scatter(x[:, 0], x[:, 1])
        plt.show()


class LinearPlant():

    def __init__(self, num_states, num_inputs=1, num_outputs=1):
        self.name = 'Linear_dim' + str(num_states)
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        sys = drss(states=num_states)
        self.A = np.array(sys.A)
        self.initial_states = None
        self.open_u = np.zeros((num_inputs,))

    def step_once(self, full_states):
        self.states = np.dot(self.A, self.states)
        return self._get_obs(full_states)

    def sim_traj(self, timesteps, full_states, stable_sample):
        self.reset()
        return np.array([self.step_once(full_states) for i in range(timesteps)])

    def _get_obs(self, full_states):
        if full_states:
            return self.states
        else:
            # return last states
            return [self.states[-1]]

    def step(self):
        self.states = np.dot(self.A, self.states)
        return self._get_obs()

    def reset(self):
        self.initial_states = np.random.randn(self.num_states,)
        self.states = self.initial_states
