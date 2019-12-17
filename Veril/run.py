import os
import argparse

from keras.models import Model, load_model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint

# from keras import regularizers

import h5py
import numpy as np

from Veril import plants
import verifier
import closed_loop
from custom_layers import JanetController
import sample_lyap
import sample_variety
from util.plots import *
# from util.plotly3d import *
# from util.samples import withinLevelSet
import time

options = {
    'plant_name': 'DoubleIntegrator',
    # 'plant_name': 'DubinsPoly',
    # 'plant_name': 'DubinsTrig',
    # 'plant_name': 'Pendulum',
    # 'plant_name': 'Satellite',
    'num_units': 4,
    'timesteps': 1000,
    'num_samples': 10000,
    'batch_size': 1,
    'epochs': 10,
    'dt': 1e-3,
    'obs_idx': None,
    'tag': '',
}


def train_RNN_controller(pre_trained=None, **kwargs):
    num_samples = kwargs.pop('num_samples')
    num_units = kwargs.pop('num_units')
    timesteps = kwargs.pop('timesteps')
    dt = kwargs.pop('dt')
    obs_idx = kwargs.pop('obs_idx')
    tag = kwargs.pop('tag')
    plant_name = kwargs.pop('plant_name')

    plant = plants.get(plant_name, dt, obs_idx)
    if pre_trained is None:
        [init_x, init_c, ext_in] = [
            Input(shape=(plant.num_states,), name='init_x'),
            Input(shape=(num_units,), name='init_c'),
            Input(shape=(None, None), name='ext_in')
        ]
        Janet_layer = JanetController(
            num_units, plant_name=plant_name, dt=dt,
            obs_idx=obs_idx)
        out = Janet_layer(ext_in, initial_state=[init_x, init_c])
        model = Model([init_x, init_c, ext_in], out)
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        print(model.summary())
    else:
        model = pre_trained

    dirname = os.path.join('/users/shenshen/Veril/data/')
    model_file_name = dirname + plant.name + '/' + \
        'unit' + str(num_units) + 'step' + str(timesteps) + tag

    callbacks = [ModelCheckpoint(model_file_name + '.h5',
                                 monitor='val_loss', verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False, mode='auto',
                                 period=1)]
    print('Train...')
    [x_train, y_train] = plant.get_data(num_samples, timesteps, num_units)
    history = model.fit(x_train, y_train, callbacks=callbacks, **kwargs)
    last_loss = history.history['loss'][-1]
    model_file_name = model_file_name + 'loss' + str(last_loss) + tag + '.h5'
    model.save(model_file_name)
    print("Saved model " + model_file_name + " to disk")


def train_V(sys_name, max_deg=3, epochs=15, method='SGD'):
    sys = closed_loop.get(sys_name)
    sym_x = sys.sym_x
    model = None
    model_dir = '../data/' + sys_name
    train_x = np.load(model_dir + '/stableSamples.npy')
    # V = sys.knownROA()
    # train_x, train_y = withinLevelSet(V)
    sys.set_features(max_deg)
    if max_deg == 3:
        loaded = np.load(model_dir + '/train_for_v_features.npz')
        [phi, dphidx, f] = [loaded['phi'], loaded['dphidx'], loaded['f']]
    else:
        [phi, dphidx, f] = sys.train_for_V_features(train_x)

    if method is 'SGD':
        y = - np.ones(phi.shape)
        if model is None:
            model = sample_lyap.poly_model_for_V(sys.num_states, max_deg)
        history = model.fit([phi, dphidx, f], y, epochs=epochs, verbose=True)
        # assert (history.history['loss'][-1] <= 0)
        P = sample_lyap.get_gram_for_V(model)
    else:
        P = verifier.solve_LP_for_V(phi, dphidx, f, num_samples=None)
    V, Vdot = sys.rescale_V(P, train_x)
    return V, Vdot, sys


def verify_via_equality(sys, V0):
    V = verifier.levelset_sos(sys, V0, do_balance=False)
    plot_funnel(V, sys.name, sys.slice)


def verify_via_variety(sys_name, init_root_threads=1, epochs=15):
    # sys = closed_loop.get(sys_name)
    # [scatterSamples(np.zeros((1, sys.num_states)), sys_name, i) for i in
    #  sys.all_slices]
    V, Vdot, sys = train_V(sys_name, epochs=epochs)
    # [plot3d(V, sys_name, i) for i in sys.all_slices]
    # verify_via_equality(sys, V)
    sys.set_sample_variety_features(V)
    Vdot = sys.sym_Vdot
    variety = sample_variety.multi_to_univariate(Vdot)
    scatterSamples(sample_variety.sample_on_variety(variety, 30), sys_name,
                   sys.slice)

    is_vanishing = False

    samples = sample_variety.sample_on_variety(variety, init_root_threads)

    while not is_vanishing:
        samples_monomial, Tinv = sample_variety.sample_monomials(
            sys, samples, variety)
        V, rho, P = sample_variety.solve_SDP_on_samples(sys, samples_monomial)
        is_vanishing, new_sample = sample_variety.check_vanishing(
            sys, variety, rho, P, Tinv)
        # samples = [np.vstack(i) for i in zip(samples, new_sample)]
        # samples = np.vstack((samples, sample_variety.sample_on_variety
        # (variety, 1)))
        samples = np.vstack((samples, new_sample))
    print(rho)
    plot_funnel(V, sys_name, sys.slice)
    return rho


def verify_via_bilinear(sys_name, max_deg=3):
    sys = closed_loop.get(sys_name)
    sys.set_features(max_deg)
    A, S = sys.do_linearization()
    V0 = sys.sym_x.T@S@sys.sym_x

    verifierOptions = verifier.opt(sys.num_states, sys.degf, do_balance=False,
                                   degV=2 * max_deg, converged_tol=1e-2, max_iterations=20)
    start = time.time()
    V = verifier.bilinear(sys.sym_x, V0, sys.sym_f, S, A, verifierOptions)
    end = time.time()
    print(end - start)
    plot_funnel(V, sys_name, sys.slice)
    return sys, V


def sim_RNN_stable_samples(**options):
    old_sampels = np.load('DIsamples.npy')
    model, model_file_name = closed_loop.get_NNorCL(NNorCL='NN', **options)
    samples = closed_loop.sample_stable_inits(
        model, 20000, 1000, lb=-1.5, ub=1.5)
    np.save('DIsamples', np.vstack([old_sampels, samples]))


# sys, V = verify_via_bilinear('VanderPol')
# sys, V = verify_via_bilinear('Pendubot',max_deg = 3)
verify_via_variety('Pendubot', init_root_threads=1, epochs=1)
# for i in range(30):
#     rho = verify_via_variety('VanderPol', init_root_threads=1, epochs=10)
#     V, Vdot, sys = train_V('VanderPol', epochs=40)
#     verify_via_equality(sys, V)


# verify_RNN_CL(max_deg=2)
# train_RNN_controller(**options)

# closed_loop.originalSysInitialV(CL)
# augedSys = closed_loop.PolyRNNCL(CL, model_file_name, taylor_approx=True)
# augedSys.do_linearization(which_dynamics='nonlinear')


############
# def verify_RNN_CL(max_deg=2):
#     CL, model_file_name = closed_loop.get_NNorCL(**options)
#     sys = closed_loop.PolyRNNCL(CL, model_file_name, taylor_approx=True)
#     sys.set_features(max_deg)
#     samples = sys.sample_init_states_w_tanh(30000, lb=-.01, ub=.01)
#     [phi, dphidx, f] = sys.train_for_V_features(samples)

#     y = np.zeros(phi.shape)
#     nx = sys.num_states
#     degf = sys.degf
#     model = sample_lyap.poly_model_for_V(nx, max_deg)
#     history = model.fit([phi, dphidx, f], y, epochs=100, shuffle=True)
#     assert (history.history['loss'][-1] <= 0)
#     P = sample_lyap.get_gram_for_V(model)
#     V0 = sys.sym_phi.T@P@sys.sym_phi
#     return V0, sys

# def SGDLevelSetGramCandidate(V, vdp, max_deg=3):
#     sym_x = vdp.sym_x
#     train_x = vdp.get_x(d=10).T
#     train_y = np.ones((train_x.shape[0], 1))
#     vdp.set_features(max_deg)
#     sigma_deg = 8
#     psi_deg = 8
#     vdp.levelset_features(V, sigma_deg)
#     [V, Vdot, xxd, psi, sigma] = vdp.get_levelset_features(train_x)
#     verifyModel = sample_lyap.gram_decomp_model_for_levelsetpoly(
#         vdp.num_states, sigma_deg, psi_deg)
#     history = verifyModel.fit([V, Vdot, xxd, psi, sigma], train_y, epochs=100,
#                               shuffle=True)
#     return verifyModel
# verifyModel = SGDLevelSetGramCandidate(V, vdp)
# [gram, g, rho, L] = sample_lyap.get_gram_trans_for_levelset_poly(verifyModel)
# # x = np.array([-1.2, 2]).reshape((1, 2))
# # pp = vdp.get_levelset_features(x)
# # print(verifyModel.predict(pp))
