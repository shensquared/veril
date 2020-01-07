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


def train_V(sys_name, max_deg=3, epochs=15, method='SGD', model=None,
            remove_one=True):
    system = closed_loop.get(sys_name)
    sym_x = system.sym_x
    model_dir = '../data/' + sys_name
    # V = system.knownROA()
    train_x = np.load(model_dir + '/stableSamples.npy')
    # train_x = np.load(model_dir + '/2-dims/stableSamples_w_2dim_zeros.npy')
    num_samples = train_x.shape[0]
    assert(train_x.shape[1]==sym_x.shape[0])
    print('x size %s' % str(train_x.shape))

    system.set_syms(max_deg, remove_one=remove_one)

    if remove_one:
        file_path = model_dir + '/train_for_v_features_remove_one' + \
            str(max_deg) + '.npz'
    else:
        file_path = model_dir + '/train_for_v_features' + str(max_deg) + '.npz'
    # train_feature_file = model_dir+'/backup-train_for_v_features3(etafull4dim).npz'
    # file_path = model_dir + '/2-dims/train_for_v_features' + \
    # str(max_deg) + '_w_2dim_zeros.npz'
    if os.path.exists(file_path):
        loaded = np.load(file_path)
        features = [loaded['phi'], loaded['eta']]
        # eval_loaded = np.load(eval_feature_file)
        # eval_features = [eval_loaded['phi'], eval_loaded['eta']]
    else:
        features = system.features_at_x(train_x)
        np.savez_compressed(file_path, phi=phi, eta=eta)
    assert(features[0].shape[0]==num_samples)
    if method is 'SGD':
        y = - np.ones((num_samples,))
        # y_eval = -np.ones((eval_x.shape[0]))
        if model is None:
            model = sample_lyap.modelV(system.num_states, max_deg, remove_one=remove_one)
            history = model.fit(features, y, epochs=epochs,
                                verbose=True, validation_split=0, shuffle=True)
            # assert (history.history['loss'][-1] <= 0)
        bad_samples, bad_predictions = eval_model(model, system, train_x, features=features)
        P = sample_lyap.get_gram_for_V(model)
        V, Vdot = system.P_to_V(P, train_x, rescale=False)
        test_model(model, system, V, Vdot, x=None)
        model_file_name = model_dir + \
            '/V_model_deg_' + str(max_deg) + '.h5'
        model.save(model_file_name)
        print("Saved model " + model_file_name + " to disk")
    else:
        P = symbolic_verifier.convexly_search_for_V_on_samples(features)
        cvx_P_filename = model_dir + '/cvx_P_' + str(max_deg) + '.npy'
        np.save(cvx_P_filename, P)
        return P
    return V, Vdot, system


def eval_model(model, system, x, features=None):
    if features is None:
        features = system.features_at_x(x)
    predicted = model.predict(features)
    bad_samples = x[predicted.ravel() > 0]
    bad_predictions = predicted[predicted.ravel() > 0]
    [scatterSamples(bad_samples, sys_name, slice_idx=i)
     for i in system.all_slices]
    return bad_samples, bad_predictions


def test_model(model, system, V, Vdot, x=None):
    if x is None:
        n_tests = 3
        test_x = np.random.randn(n_tests, system.num_states)
    else:
        test_x = x
    test_features = system.features_at_x(test_x)
    test_prediction = model.predict(test_features)
    test_V = system.get_v_values(test_x,V=V)
    test_Vdot = system.get_v_values(test_x,V=Vdot)
    return [test_prediction,test_V,test_Vdot]



def verify_via_equality(system, V):
    if V is None:
        A, S = system.linearized_A_and_P()
        V = system.sym_x.T@S@system.sym_x
    V = symbolic_verifier.levelset_sos(system, V, do_balance=False)
    plot_funnel(V, system.name, system.slice, add_title=' - Equality ' +
                'Constrainted Result')


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
    plot_funnel(V, sys_name, system.slice, add_title=' - Sampling Variety ' +
                'Result')
    return rho


def verify_via_bilinear(sys_name, max_deg=3):
    system = closed_loop.get(sys_name)
    system.set_syms(max_deg)
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
    plot_funnel(V, sys_name, system.slice, add_title=' - Bilinear Result')
    return system, V


sys_name = 'VanderPol'
# sys_name = 'Pendubot'
epochs = 20
max_deg = 3
init_root_threads = 30

system = closed_loop.get(sys_name)
# model = sample_lyap.get_V_model(sys_name, max_deg)
model = None
# system.scatter_stable_samples()

# test_model(system, model = None)

# system, V  = verify_via_bilinear(sys_name, max_deg=4)
V, Vdot, system = train_V(sys_name, epochs=epochs, max_deg=max_deg,
                          model=model, remove_one=True)
# [plot_funnel(V*3, sys_name, i) for i in system.all_slices]
[plot3d(V, sys_name, i, level_sets=True) for i in system.all_slices]
[plot3d(Vdot, sys_name, i, level_sets=False, r_max=1.6)
 for i in system.all_slices]
verify_via_equality(system, V)
# for i in range(30):
#     verify_via_variety(system, V, init_root_threads=init_root_threads)


############
# Dirty code below, but may be useful for refrence
# def SGDLevelSetGramCandidate(V, vdp, max_deg=3):
#     sym_x = vdp.sym_x
#     train_x = vdp.get_x(d=10).T
#     train_y = np.ones((train_x.shape[0], 1))
#     vdp.set_syms(max_deg)
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
