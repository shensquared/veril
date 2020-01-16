import time
import numpy as np
# import argparse
from veril.systems import get_system
from veril.sample_lyap import get_V

from veril.symbolic_verifier import verify_via_equality, verify_via_bilinear
from veril.sample_variety import verify_via_variety
from veril.util.plots import *


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




sys_name = 'VanderPol'
# sys_name = 'Pendubot'
# sys_name = 'PendulumTrig'
# sys_name = 'PendulumRecast'

# train_or_load = 'Train'
train_or_load = 'Load'

degFeatures = 3
degU = 2
epochs = 10
remove_one = True

init_root_threads = 25


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
