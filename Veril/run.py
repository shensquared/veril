import os
import argparse

from keras.models import Model, load_model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.utils import CustomObjectScope
from keras import backend as K
# from keras import regularizers

import h5py
import numpy as np

# import scipy.io

import Plants
from CustomLayers import JanetController
from Verifier import *


num_units = 4
plant_name = "DoubleIntegrator"
# plant_name = "Satellite"
# plant_name = "Pendulum"
timesteps = 1000
NNorCL = 'CL'


def train(plant_name=None, num_units=4, timesteps=100,
          num_samples=10000, batch_size=1, epochs=3, dt=1e-3, obs_idx=None,
          tag='', pre_trained=None):
    plant = Plants.get(plant_name, dt, obs_idx)
    if pre_trained is None:
        [init_x, init_c, ext_in] = [Input(shape=(plant.num_states,), name='init_x'),
                                    Input(shape=(num_units,), name='init_c'),
                                    Input(shape=(timesteps, None), name='ext_in')]
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
        'unit' + str(num_units) + 'step' + str(timesteps) + tag + '.h5'

    callbacks = [ModelCheckpoint(model_file_name,
                                 monitor='val_loss', verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False, mode='auto',
                                 period=1)]
    print('Train...')
    [x_train, y_train] = plant.get_data(num_samples, timesteps, num_units)
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs, callbacks=callbacks)
    last_loss = history.history['loss'][-1]
    model_file_name = dirname + plant.name + '/' + \
        'unit' + str(num_units) + 'step' + str(timesteps) + 'loss' + \
        str(last_loss) + tag + '.h5'
    model.save(model_file_name)
    print("Saved model " + model_file_name + " to disk")


def get_NNorCL(num_units, plant_name, timesteps, tag='', NNorCL='CL'):
    # dirname = os.path.dirname(__file__)
    dirname = os.path.join('/users/shenshen/Veril/data/')
    model_file_name = dirname + plant_name + '/' + \
        'unit' + str(num_units) + 'step' + str(timesteps) + tag + '.h5'

    with CustomObjectScope({'JanetController': JanetController}):
        model = load_model(model_file_name)
    print(model.summary())
    if NNorCL is 'NN':
        return model
    elif NNorCL is 'CL':
        for this_layer in model.layers:
            if hasattr(this_layer, 'cell'):
                return this_layer


CL = get_NNorCL(num_units, plant_name, timesteps, NNorCL='CL')
originalSysInitialV(CL)
# [x,f] = augDynamics(CL)
# linearizeAugDynamics(x,f)

# NN = get_NNorCL(num_units, plant_name, timesteps, NNorCL='NN')
# train(pre_trained=NN, plant_name=plant_name, num_units=num_units,
# timesteps=timesteps, batch_size=1,epochs=3)

# train(pre_trained=None, plant_name=plant_name, num_units=num_units,
#       timesteps=timesteps, batch_size=10, epochs=3)
