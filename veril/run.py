import os
import time
import numpy as np
# import argparse
from keras.callbacks import ModelCheckpoint, EarlyStopping
from veril.systems import get_system

from veril import symbolic_verifier, sample_lyap, sample_variety
from veril.util.plots import *


def load_features_dataset(file_path, loop_closed):
    l = np.load(file_path)
    if loop_closed:
        features = [l['phi'], l['eta']]
    else:
        features = [l['g'], l['phi'], l['dphidx'], l['ubasis']]
    return features


def get_V(system, train_or_load, **kwargs):
    degFeatures = system.degFeatures
    remove_one = system.remove_one
    loop_closed = system.loop_closed

    tag = '_degV' + str(2 * degFeatures)
    model_dir = '../data/' + system.name

    if loop_closed and remove_one:
        tag = tag + '_rm'
    if not loop_closed:
        tag = tag + 'degU' + str(degU)

    if train_or_load is 'Train':
        nx = system.num_states

        if loop_closed:
            train_x = np.load(model_dir + '/stableSamples.npy')
            n_samples = train_x.shape[0]
            assert(train_x.shape[1] == nx)
            print('x size %s' % str(train_x.shape))
        else:
            train_x = None

        file_path = model_dir + '/features' + tag + '.npz'
        if os.path.exists(file_path):
            features = load_features_dataset(file_path, loop_closed)
        else:
            features = system.features_at_x(train_x, file_path)

        n_samples = features[0].shape[0]
        y = - np.ones((n_samples,))
        model = sample_lyap.model_V(system)
        history = model.fit(features, y, **kwargs)
        # assert (history.history['loss'][-1] <= 0)
        # bad_samples, bad_predictions = eval_model(
        #     model, system, train_x, features=features)
        model_file_name = model_dir + '/V_model' + tag + '.h5'
        model.save(model_file_name)
        print('Saved model ' + model_file_name + ' to disk')

    elif train_or_load is 'Load':
        # TODO: decide if should save the model or directly save V
        train_x = None
        model = sample_lyap.get_V_model(model_dir, tag)

    P, u_weights = sample_lyap.get_model_weights(model, loop_closed)
    if not loop_closed:
        system.close_the_loop(u_weights)
    V, Vdot = system.P_to_V(P, samples=None)
    # test_model(model, system, V, Vdot, x=None)
    return V, Vdot, system


def eval_model(model, system, x, features=None):
    if features is None:
        features = system.features_at_x(x)
    predicted = model.predict(features)
    bad_samples = x[predicted.ravel() > 0]
    bad_predictions = predicted[predicted.ravel() > 0]
    [scatterSamples(bad_samples, system, slice_idx=i)
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
    test_V = system.get_v_values(test_x, V=V)
    test_Vdot = system.get_v_values(test_x, V=Vdot)
    return [test_prediction, test_V, test_Vdot]


def verify_via_equality(system, V0):
    if V0 is None:
        A, S, V0 = system.linearized_quadractic_V()
    start = time.time()
    V = symbolic_verifier.levelset_sos(system, V0, do_balance=False)
    end = time.time()
    print('equlity constrained time %s' % (end - start))
    plot_funnel(V, system, slice_idx=system.slice, add_title=' - Equality ' +
                'Constrainted Result')


def verify_via_variety(system, V, init_root_threads=1):
    system.set_sample_variety_features(V)
    Vdot = system.sym_Vdot
    variety = sample_variety.multi_to_univariate(Vdot)

    is_vanishing = False

    samples = sample_variety.sample_on_variety(variety, init_root_threads)

    while not is_vanishing:
        samples_monomial, Tinv = sample_variety.sample_monomials(
            system, samples, variety)
        start = time.time()
        V, rho, P = sample_variety.solve_SDP_on_samples(
            system, samples_monomial)
        is_vanishing, new_sample = sample_variety.check_vanishing(
            system, variety, rho, P, Tinv)
        end = time.time()
        print('sampling variety time %s' % (end - start))

        # samples = [np.vstack(i) for i in zip(samples, new_sample)]
        # samples = np.vstack((samples, sample_variety.sample_on_variety
        # (variety, 1)))
        samples = np.vstack((samples, new_sample))
    print(rho)
    plot_funnel(V / rho, system, slice_idx=system.slice, add_title=' - Sampling Variety '
                + 'Result')
    return rho


def verify_via_bilinear(system, **kwargs):
    sys_name = system.name
    degV = system.degV
    degf = system.degf
    degL = degV - 1 + degf
    options = {'degV': degV, 'do_balance': False, 'degL1': degL, 'degL2': degL,
               'converged_tol': 1e-2, 'max_iterations': 20}

    start = time.time()
    A, S, V0 = system.linearized_quadractic_V()

    if 'V0' in kwargs:
        V0 = kwargs['V0']
    V = symbolic_verifier.bilinear(
        system.sym_x, V0, system.sym_f, S, A, **options)
    end = time.time()
    print('bilinear time %s' % (end - start))
    plot_funnel(V, system, slice_idx=system.slice,
                add_title=' - Bilinear Result')
    return system, V


def cvx_V(sys_name, degFeatures, remove_one=False):
    tag = str(degFeatures)
    system = get_system(sys_name, degFeatures, remove_one=remove_one)
    model_dir = '../data/' + sys_name
    # train_x = np.load(model_dir + '/stableSamples.npy')
    num_samples = train_x.shape[0]
    assert(train_x.shape[1] == system.num_states)
    print('x size %s' % str(train_x.shape))

    file_path = model_dir + '/features_' + tag + '.npz'

    if os.path.exists(file_path):
        l = np.load(file_path)
        features = [l['phi'], l['eta']]
    else:
        features = system.features_at_x(train_x)
        np.savez_compressed(file_path, phi=features[0], eta=features[1])
    assert(features[0].shape[0] == num_samples)
    P = symbolic_verifier.convexly_search_for_V_on_samples(features)
    cvx_P_filename = model_dir + '/cvx_P_' + str(degFeatures) + '.npy'
    np.save(cvx_P_filename, P)
    V, Vdot = system.P_to_V(P)
    return V, Vdot, system


# sys_name = 'VanderPol'
# sys_name = 'Pendubot'
sys_name = 'PendulumTrig'


train_or_load = 'Train'
# train_or_load = 'Load'

degFeatures = 3
degU = 2
epochs = 3
remove_one = True

init_root_threads = 100


system = get_system(sys_name, degFeatures, degU, remove_one=remove_one)
# stableSamples = system.sample_stable_inits(d=1, num_grid=10)

V, Vdot, system = get_V(system, train_or_load, epochs=epochs,
                        verbose=True, validation_split=0, shuffle=True)

verify_via_equality(system, V)

[plot3d(V, i, level_sets=True) for i in system.all_slices]
[plot3d(Vdot, i, level_sets=False, r_max=1.6)
 for i in system.all_slices]

initial = np.random.uniform(-np.pi, np.pi, (4, 2))
plot_traj(initial, system, int_horizon=20, slice_idx=0)

# verify_via_bilinear(system, V0=V)
# verify_via_variety(system, V, init_root_threads=init_root_threads)

############
# Dirty code below, but may be useful for refrence
# def SGDLevelSetGramCandidate(V, vdp, degFeatures=3):
#     sym_x = vdp.sym_x
#     train_x = vdp.get_x(d=10).T
#     train_y = np.ones((train_x.shape[0], 1))
#     vdp.set_syms(degFeatures)
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
