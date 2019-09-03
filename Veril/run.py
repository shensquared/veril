import os
import argparse

from keras.models import Model, load_model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.utils import CustomObjectScope
# from keras import regularizers

import h5py
import scipy.io
import numpy as np

import Plants, CustomLayers
from CustomLayers import JanetController

def train(num_units=4, plant_name='Pendulum', timesteps=100,
          num_samples=10000, batch_size=1, epochs=3, dt=1e-3, obs_idx=None,
          tag=''):
    plant = Plants.get(plant_name, dt, obs_idx)
    [init_x, init_c, ext_in] = [Input(shape=(plant.num_states,), name='init_x'),
                                Input(shape=(num_units,), name='init_c'),
                                Input(shape=(timesteps, None), name='ext_in')]
    Janet_layer = JanetController(
        num_units, plant_name=plant_name, dt=dt, obs_idx=obs_idx)
    out = Janet_layer(ext_in, initial_state=[init_x, init_c])
    model = Model([init_x, init_c, ext_in], out)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    print(model.summary())

    # dirname = os.path.dirname(__file__)
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
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, callbacks=callbacks)

    model.save(model_file_name)
    print("Saved model " + model_file_name + " to disk")

train()