import os
import time
import numpy as np
# import argparse

from keras.models import Model, load_model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py

from veril import symbolic_verifier, systems, sample_lyap, sample_variety
from veril.util.plots import *
# from util.samples import withinLevelSet


def get_system(sys_name, degFeatures, degU, remove_one=True):
    system = systems.get(sys_name)
    system.set_syms(degFeatures, degU, remove_one=remove_one)
    return system


def getUV(system, train_or_load, **kwargs):
    sys_name = system.name
    degFeatures = system.degFeatures
    degU = system.degU
    tag = '_degV' + str(2 * degFeatures) + 'degU' + str(degU)

    if train_or_load is 'Train':
        nx = system.num_states
        model_dir = '../data/' + sys_name
        file_path = model_dir + '/features' + tag + '.npz'
        if os.path.exists(file_path):
            loaded = np.load(file_path)
            features = [loaded['g'], loaded['phi'], loaded['dphidx'], loaded
                        ['ubasis']]
        else:
            features = system.features_at_x()
            np.savez_compressed(file_path, g=features[0], phi=features[1],
                                dphidx=features[2], ubasis=features[3])
        n_samples = features[0].shape[0]
        y = - np.ones((n_samples,))
        model = sample_lyap.model_V(system)
        history = model.fit(features, y, **kwargs)
        model_file_name = model_dir + '/V_model' + tag + '.h5'
        model.save(model_file_name)
        print('Saved model ' + model_file_name + ' to disk')

    elif train_or_load is 'Load':
        # TODO: decide if should save the model or directly save V
        train_x = None
        model = sample_lyap.get_V_model(sys_name, tag)

    P, u_weights = sample_lyap.get_model_weights(model)
    system.close_the_loop(u_weights)
    V, Vdot = system.P_to_V(P, samples=None)
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


sys_name = 'PendulumTrig'
# train_or_load = 'Train'
train_or_load = 'Load'

degFeatures = 3
degU = 2
# init_root_threads = 18
epochs = 7
remove_one = True

system = get_system(sys_name, degFeatures, degU)
V, Vdot, system = getUV(system, train_or_load, epochs=epochs, verbose=True,
                        validation_split=0, shuffle=True)
stableSamples = system.sample_stable_inits(d=1, num_grid=10)
np.save('stableSamples', stableSamples)
scatterSamples(stableSamples, system, slice_idx = system.slice_idx)

# [plot_funnel(V, system, slice_idx = i) for i in system.all_slices]
[plot3d(V, i, level_sets=True) for i in system.all_slices]
[plot3d(Vdot, i, level_sets=False, r_max=1.6)
 for i in system.all_slices]
