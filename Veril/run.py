import os
import argparse

from keras.models import Model, load_model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint

# from keras import regularizers

import h5py
import numpy as np

import Plants
import Verifier
import ClosedControlledLoop
from CustomLayers import JanetController


options = {
    'plant_name': 'DoubleIntegrator',
    # 'plant_name': 'Pendulum',
    # 'plant_name': 'Satellite',
    'num_units': 4,
    'timesteps': 1000,
    'num_samples': 100,
    'batch_size': 1,
    'epochs': 3,
    'dt': 1e-3,
    'obs_idx': None,
    'tag': '',
}


def train(pre_trained=None, **kwargs):

    num_samples = kwargs.pop('num_samples')
    num_units = kwargs.pop('num_units')
    timesteps = kwargs.pop('timesteps')
    dt = kwargs.pop('dt')
    obs_idx = kwargs.pop('obs_idx')
    tag = kwargs.pop('tag')
    plant_name = kwargs.pop('plant_name')

    plant = Plants.get(plant_name, dt, obs_idx)
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

# CL = ClosedControlledLoop.get_NNorCL(**options)
# ClosedControlledLoop.originalSysInitialV(CL)
# augedSys = ClosedControlledLoop.augmentedTanhPolySys(CL)
# x,f = augedSys.symbolicStatesAndDynamics()
# samples = augedSys.sampleInitialStatesInclduingTanh(3)
# print(samples)
# A,S = augedSys.linearizeAugmentedTanhPolySys()

# NN = ClosedControlledLoop.get_NNorCL(NNorCL='NN', **options)
train(pre_trained=None, **options)
