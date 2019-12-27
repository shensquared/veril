import os
import time
import numpy as np
# import argparse

from keras.models import Model, load_model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py

from veril import symbolic_verifier, closed_loop, sample_lyap, sample_variety
from veril.util.plots import *
# from util.samples import withinLevelSet


def train_V(sys_name, max_deg=3, epochs=15, method='SGD'):
    system = closed_loop.get(sys_name)
    sym_x = system.sym_x
    model = None
    model_dir = '../data/' + sys_name
    # V = system.knownROA()
    train_x = np.load(model_dir + '/stableSamples.npy')
    system.set_features(max_deg)

    file_path = model_dir + '/train_for_v_features' + str(max_deg) + '.npz'
    if os.path.exists(file_path):
        loaded = np.load(file_path)
        [phi, eta] = [loaded['phi'], loaded['eta']]
    else:
        [phi, eta] = system.train_for_V_features(train_x)

    if method is 'SGD':
        num_samples = phi.shape[0]
        y = - np.ones((num_samples,))
        if model is None:
            model = sample_lyap.poly_model_for_V(system.num_states, max_deg)
        history = model.fit([phi, eta], y, epochs=epochs, verbose=True)
        # assert (history.history['loss'][-1] <= 0)
        model_file_name = model_dir + '/V_model_deg_' + str(max_deg) + '.h5'
        model.save(model_file_name)
        print("Saved model " + model_file_name + " to disk")
        P = sample_lyap.get_gram_for_V(model)
    else:
        P = symbolic_verifier.convexly_search_for_V_on_samples([phi, eta])
        cvx_P_filename = model_dir + '/cvx_P_' + str(max_deg) + '.npy'
        np.save(cvx_P_filename, P)

    predicted = model.predict([phi, eta])
    evals = model.evaluate([phi,eta], y)
    V, Vdot = system.rescale_V(P, train_x)
    # test_x = np.random.randn(1,2)
    # test = system.train_for_V_features(test_x)
    # print('model predicted')
    # print(model.predict(test))
    return V, Vdot, system


def verify_via_equality(system, V):
    V = symbolic_verifier.levelset_sos(system, V, do_balance=False)
    plot_funnel(V, system.name, system.slice, add_title='')


def verify_via_variety(system, V, init_root_threads=1):
    system.set_sample_variety_features(V)
    Vdot = system.sym_Vdot
    variety = sample_variety.multi_to_univariate(Vdot)
    # [scatterSamples(sample_variety.sample_on_variety(variety, 30, slice_idx
    #    =i), sys_name, i) for i in system.all_slices]

    is_vanishing = False

    samples = sample_variety.sample_on_variety(variety, init_root_threads)

    while not is_vanishing:
        samples_monomial, Tinv = sample_variety.sample_monomials(
            system, samples, variety)
        V, rho, P = sample_variety.solve_SDP_on_samples(
            system, samples_monomial)
        is_vanishing, new_sample = sample_variety.check_vanishing(
            system, variety, rho, P, Tinv)
        # samples = [np.vstack(i) for i in zip(samples, new_sample)]
        # samples = np.vstack((samples, sample_variety.sample_on_variety
        # (variety, 1)))
        samples = np.vstack((samples, new_sample))
    print(rho)
    plot_funnel(V, sys_name, system.slice)
    return rho


def verify_via_bilinear(sys_name, max_deg=3):
    system = closed_loop.get(sys_name)
    system.set_features(max_deg)
    A, S = system.linearized_A_and_P()
    V = system.sym_x.T@S@system.sym_x

    verifierOptions = symbolic_verifier.opt(system.num_states, system.degf,
                                            do_balance=False, degV=2 * max_deg,
                                            converged_tol=1e-2,
                                            max_iterations=20)
    start = time.time()
    V = symbolic_verifier.bilinear(
        system.sym_x, V, system.sym_f, S, A, verifierOptions)
    end = time.time()
    print('bilinear time %s' % (end - start))
    plot_funnel(V, sys_name, system.slice)
    return system, V


sys_name = 'VanderPol'
# sys_name = 'Pendubot'


system = closed_loop.get(sys_name)
# system.scatter_stable_samples()
epochs =20
max_deg = 3
init_root_threads = 22

# V, Vdot, system = train_V(sys_name, epochs=epochs, max_deg=max_deg)
# model = sample_lyap.get_V_model(sys_name, max_deg)
# [plot3d(V, sys_name, i, level_sets=True) for i in system.all_slices]
V, Vdot, system = train_V(sys_name, epochs=epochs, max_deg=max_deg,
                          method='SGD')
verify_via_equality(system, V)
for i in range(30):
    verify_via_variety(system, V, init_root_threads=init_root_threads)

# system, V = verify_via_bilinear(sys_name)


############
# Dirty code below, but may be useful for refrence
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
