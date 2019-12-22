from keras import backend as K
from keras.utils import CustomObjectScope
from keras.models import load_model

from veril import plants
from veril.custom_layers import JanetController


def get_NNorCL(NNorCL='CL', **kwargs):
    num_samples = kwargs['num_samples']
    num_units = kwargs['num_units']
    timesteps = kwargs['timesteps']
    dt = kwargs['dt']
    obs_idx = kwargs['obs_idx']
    tag = kwargs['tag']
    plant_name = kwargs['plant_name']

    dirname = os.path.dirname(__file__)
    # dirname = os.path.join('/users/shenshen/veril/data/')
    dirname = dirname + '../data/'
    model_file_name = dirname + plant_name + '/' + \
        'unit' + str(num_units) + 'step' + str(timesteps) + tag

    with CustomObjectScope({'JanetController': JanetController}):
        model = load_model(model_file_name + '.h5')
    print(model.summary())
    if NNorCL is 'NN':
        return model, model_file_name
    elif NNorCL is 'CL':
        for this_layer in model.layers:
            if hasattr(this_layer, 'cell'):
                return [this_layer, model_file_name]


def call_CLsys(CL, tm1, num_samples):
    inputs = K.placeholder()
    states = [K.placeholder(shape=(num_samples, CL.cell.num_plant_states)),
              K.placeholder(shape=(num_samples, CL.cell.units))]
    [x_tm2, c_tm2] = CL.cell.call(inputs, states, training=False)[1]
    feed_dict = dict(zip(states, tm1))
    sess = K.get_session()
    x_tm2 = sess.run(x_tm2, feed_dict=feed_dict)
    c_tm2 = sess.run(c_tm2, feed_dict=feed_dict)
    return [x_tm2, c_tm2]


def batchSim(CL, timesteps, num_samples=10000, init=None):
    """return two sets of initial conditions based on the simulated results.
    One set is the stable trajectory and the other set is the unstable one.

    Args:
    """
    if init is None:
        x = np.random.randn(num_samples, CL.cell.num_plant_states)
        c = np.zeros((num_samples, CL.cell.units))
        init = [x, c]

    for i in range(timesteps):
        init = call_CLsys(CL, init, num_samples)
    return init


def sample_stable_inits(model, num_samples, timesteps, **kwargs):
    for this_layer in model.layers:
        if hasattr(this_layer, 'cell'):
            CL = this_layer
    plant = plants.get(CL.plant_name, CL.dt, CL.obs_idx)
    init = plant.get_data(num_samples, timesteps,
                          CL.units, random_c=True, **kwargs)[0]
    pred = model.predict(init)
    pred_norm = np.sum(np.abs(pred)**2, axis=-1)**(1. / 2)
    stable_init = [i[np.isclose(pred_norm, 0)] for i in init]
    # don't need the external disturbance for now
    return np.hstack(stable_init[:-1])


class PolyRNNCL(ClosedLoopSys):
    """Summary
    # returns the CONTINUOUS TIME closed-loop dynamics of the augmented states,
    # which include the plant state x, the RNN state c, the added two states
    # from the tanh nonlinearity, tau_c, and tau_f
    """

    def __init__(self, CL, model_file_name, taylor_approx=False):
        self.output_kernel = (K.eval(CL.cell.output_kernel))
        self.feedthrough_kernel = (K.eval(CL.cell.feedthrough_kernel))
        self.recurrent_kernel_f = (K.eval(CL.cell.recurrent_kernel_f))
        self.kernel_f = (K.eval(CL.cell.kernel_f))
        self.recurrent_kernel_c = (K.eval(CL.cell.recurrent_kernel_c))
        self.kernel_c = (K.eval(CL.cell.kernel_c))

        self.plant = plants.get(CL.plant_name, CL.dt, CL.obs_idx)
        self.nx = CL.cell.num_plant_states
        self.units = CL.units
        self.dt = CL.dt

        self.taylor_approx = taylor_approx

        prog = MathematicalProgram()
        self.x = prog.NewIndeterminates(self.nx, "x")
        self.c = prog.NewIndeterminates(self.units, "c")

        if taylor_approx:
            self.sym_x = np.concatenate((self.x, self.c))
            self.num_states = self.nx + self.units
        else:
            self.tau_f = prog.NewIndeterminates(self.units, "tf")
            self.tau_c = prog.NewIndeterminates(self.units, "tc")
            self.sym_x = np.concatenate(
                (self.x, self.c, self.tau_f, self.tau_c))
            self.num_states = self.nx + 3 * self.units
        self.model_file_name = model_file_name

    def inverse_recast_map(self):
        [arg_f, arg_c] = self.args_for_tanh(self.x, self.c)
        tau_f = [tanh(i) for i in arg_f]
        tau_c = [tanh(i) for i in arg_c]
        env = dict(zip(np.append(self.tau_c, self.tau_f), tau_c + tau_f))
        return env

    def args_for_tanh(self, x, c):
        arg_f = x@self.kernel_f + c@self.recurrent_kernel_f
        arg_c = x@self.kernel_c + c@self.recurrent_kernel_c
        return [arg_f, arg_c]

    def train_for_V_features(self, x):
        # x: (num_samples, sys_dim)
        f = self.nonlinear_dynamics(sample_states=x)
        self.verifi_f = self.nonlinear_dynamics()
        n_samples = x.shape[0]
        phi = np.zeros((n_samples, self.sym_phi.shape[0]))
        dphidx = np.zeros((n_samples, self.sym_phi.shape[0], self.num_states))
        for i in range(n_samples):
            env = dict(zip(self.sym_x, x[i, :]))
            phi[i, :] = [i.Evaluate(env) for i in self.sym_phi]
            dphidx[i, :, :] = [[i.Evaluate(env) for i in j]for j in
                               self.sym_dphidx]
        # np.savez(self.model_file_name + '-' + str(n_samples) +
        #          'samples', phi=phi, dphidx=dphidx, f=f)
        return [phi, dphidx, f]

    def nonlinear_dynamics(self, sample_states=None):
        if sample_states is None:
            x = self.x
            c = self.c
        else:
            x = sample_states[:, 0:self.nx]
            c = sample_states[:, self.nx:self.nx + self.units]

        shift_y_tm1 = self.plant.get_obs(x) - self.plant.y0
        u = c@self.output_kernel + shift_y_tm1@self.feedthrough_kernel
        xdot = self.plant.xdot(x.T, u.T).T
        ydot = self.plant.ydot(x.T, u.T).T
        # use the argument to approximate the tau:
        [arg_f, arg_c] = self.args_for_tanh(shift_y_tm1, c)
        if sample_states is None:
            tau_f = np.array([tanh(i) for i in arg_f])
            tau_c = np.array([tanh(i) for i in arg_c])
        else:
            tau_f = np.tanh(arg_f)
            tau_c = np.tanh(arg_c)
        cdot = (.5 * (- c + c * tau_f + tau_c - tau_c * tau_f)) / self.dt
        # TODO: should be y0dot but let's for now assume the two are the same
        # (since currently all y0=zeros)
        f = np.hstack((xdot, cdot))
        return f

    def polynomial_dynamics(self, sample_states=None):
        if self.taylor_approx:
            if sample_states is None:
                x = self.x
                c = self.c
            else:
                x = sample_states[:, 0:self.nx]
                c = sample_states[:, self.nx:self.nx + self.units]

            shift_y_tm1 = self.plant.get_obs(x) - self.plant.y0
            u = c@self.output_kernel + shift_y_tm1@self.feedthrough_kernel
            xdot = self.plant.xdot(x.T, u.T).T
            ydot = self.plant.ydot(x.T, u.T).T
            # use the argument to approximate the tau:
            [tau_f, tau_c] = self.args_for_tanh(shift_y_tm1, c)
            cdot = (.5 * (- c + c * tau_f + tau_c - tau_c * tau_f)) / self.dt
            # TODO: should be y0dot but let's for now assume the two are the same
            # (since currently all y0=zeros)
            f = np.hstack((xdot, cdot))
            return f
        else:
            if sample_states is None:
                x = self.x
                c = self.c
                tau_f = self.tau_f
                tau_c = self.tau_c
            else:
                x = sample_states[:, 0:self.nx]
                c = sample_states[:, self.nx:self.nx + self.units]
                tau_f = sample_states[:, self.nx +
                                      self.units:self.nx + 2 * self.units]
                tau_c = sample_states[:, self.nx + 2 *
                                      self.units:self.nx + 3 * self.units]

            shift_y_tm1 = self.plant.get_obs(x) - self.plant.y0

            u = c@self.output_kernel + shift_y_tm1@self.feedthrough_kernel
            xdot = self.plant.xdot(x.T, u.T).T
            ydot = self.plant.ydot(x.T, u.T).T
            cdot = (.5 * (- c + c * tau_f + tau_c - tau_c * tau_f)) / self.dt
            # TODO: should be y0dot but let's for now assume the two are the same
            # (since currently all y0=zeros)
            tau_f_dot = (1 - tau_f**2) * self.args_for_tanh(ydot, cdot)[0]
            tau_c_dot = (1 - tau_c**2) * self.args_for_tanh(ydot, cdot)[1]
            f = np.hstack((xdot, cdot, tau_f_dot, tau_c_dot))
            return f

    def sample_init_states_w_tanh(self, num_samples, **kwargs):
        """sample initial states in [x,c,tau_f,tau_c] space. But since really only
        x and c are independent, bake in the relationship that tau_f=tanh(arg_f)
        and tau_c=tanh(arg_c) here. Also return the augmented poly system dyanmcis
        f for the downstream verification analysis.

        Args:
            CL (TYPE): closedcontrolledsystem

        Returns:
            TYPE: Description
        """
        [x, _, _] = self.plant.get_data(
            num_samples, 1, self.units, **kwargs)[0]
        shifted_y = self.plant.get_obs(x) - self.plant.y0
        c = np.random.uniform(-.01, .01, (num_samples, self.units))
        if self.taylor_approx:
            s = np.hstack((x, c))
        else:
            tanh_f = np.tanh(self.args_for_tanh(shifted_y, c)[0])
            tanh_c = np.tanh(self.args_for_tanh(shifted_y, c)[1])
            s = np.hstack((x, c, tanh_f, tanh_c))
        return s

    def do_linearization(self, which_dynamics='nonlinear'):
        """
        linearize f, which is the augmented (via the change of variable recasting)
        w.r.t. the states x.

        Args:
            x (TYPE): States
            f (TYPE): the augmented dynamics

        Returns:
            TYPE: the linearization, evaluated at zero
        """
        # TODO: for now linearize at zero, need to linearize at plant.x0

        if which_dynamics is 'nonlinear':
            x = self.x
            c = self.c
            xc = np.concatenate((self.x, self.c))
            f = self.nonlinear_dynamics()
        elif which_dynamics is 'polynomial':
            xc = self.sym_x
            f = self.polynomial_dynamics()
        J = Jacobian(f, xc)
        env = dict(zip(xc, np.zeros(xc.shape)))

        A = np.array([[i.Evaluate(env) for i in j]for j in J])
        # print('A  %s' % A)
        print('eig of the linearized A matrix for augmented with tanh poly system %s' % (
            eig(A)[0]))
        S = solve_lyapunov(A.T, -np.eye(xc.shape[0]))
        return A, S


def originalSysInitialV(CL):
    plant = plants.get(CL.plant_name, CL.dt, CL.obs_idx)
    A0 = CL.linearize()
    full_dim = plant.num_states + CL.units
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(full_dim, "x")
    if not plant.manifold:
        S0 = solve_lyapunov(A0.T, -np.eye(full_dim))
    else:
        P = prog.NewSymmetricContinuousVariables(full_dim, "P")
        prog.AddPositiveSemidefiniteConstraint(P)
        prog.AddPositiveSemidefiniteConstraint(P + P.T)
        # V = x.T@P@x
        Vdot = x.T@P@A0@x + x.T@A0.T@P@x
        r = prog.NewContinuousVariables(1, "r")[0]
        Vdot = Vdot + r * plant.get_manifold(x)
        slack = prog.NewContinuousVariables(1, "s")[0]
        prog.AddConstraint(slack >= 0)

        prog.AddSosConstraint(-Vdot - slack * x.T@np.eye(full_dim)@x)
        prog.AddCost(-slack)
        solver = MosekSolver()
        solver.set_stream_logging(False, "")
        result = solver.Solve(prog, None, None)
        # print(result.get_solution_result())
        assert result.is_success()
        slack = result.GetSolution(slack)
        print('slack is %s' % (slack))
        S0 = result.GetSolution(P)
    print('eig of orignal A  %s' % (eig(A0)[0]))
    print('eig of orignal SA+A\'S  %s' % (eig(A0.T@S0 + S0@A0)[0]))
    return x.T@S0@x
