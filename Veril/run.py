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

import Plants
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

def load_cust_model(num_units, plant_name, timesteps, tag=''):
    # dirname = os.path.dirname(__file__)
    dirname = os.path.join('/users/shenshen/Veril/data/')
    model_file_name = dirname + plant_name + '/' + \
        'unit' + str(num_units) + 'step' + str(timesteps) + tag + '.h5'

    with CustomObjectScope({'JanetController': JanetController}):
        model = load_model(model_file_name)
    print(model.summary())
    for this_layer in model.layers:
        if hasattr(this_layer, 'cell'):
            inputs=None,
            states=[np.array([0, -1, 0]),np.zeros(num_units,)]
            # sys = CLoop(this_layer, timesteps, tag)
            x_tm2, [x_tm2, c_tm2] = this_layer.cell.call(inputs,states,training=False)
    return [x_tm2, c_tm2]
load_cust_model(4, "Pendulum", 100, tag='')

class CLoop(object):

    def __init__(self, this_layer, timesteps, tag):
        self.steps = timesteps
        self.tag = tag
        self.plant_name = this_layer.plant_name
        self.dt = this_layer.dt
        self.obs_idx = this_layer.obs_idx
        self.units = this_layer.units
        self.use_bias = this_layer.use_bias
        this_weights = this_layer.get_weights()
        # y kernel
        kernel = this_weights[0]
        self.kernel_f = kernel[:, :self.units]
        self.kernel_c = kernel[:, self.units * 1: self.units * 2]
        # c kernel
        recurrent_kernel = this_weights[1]
        self.recurrent_kernel_f = recurrent_kernel[:, :self.units]
        self.recurrent_kernel_c = recurrent_kernel[:, self.units:
                                                   self.units * 2]
        # to output u kernel
        self.output_kernel = this_weights[2]
        # bias
        if this_layer.use_bias:
            bias = this_weights[3]
            self.bias_f = bias[:self.units]
            self.bias_c = bias[self.units: self.units * 2]
        # external input kernel
            if this_layer.external_input:
                input_kernel = this_weights[4]
        else:
            if this_layer.external_input:
                input_kernel = this_weights[3]

    def call(self, states):
        plant = Plants.get(self.plant_name, self.dt, self.obs_idx)

        x_tm1 = states[0].reshape(-1)
        c_tm1 = states[1].reshape(-1)  # previous cell memory state
        y_tm1 = plant.np_get_obs(x_tm1)
        shift_y_tm1 = y_tm1 - plant.y0

        c_tm1_f = c_tm1
        c_tm1_c = c_tm1

        c_f = np.dot(c_tm1_f, self.recurrent_kernel_f)
        c_c = np.dot(c_tm1_c, self.recurrent_kernel_c,)

        # adding the plant output feedback
        c_f = c_f + np.dot(shift_y_tm1, self.kernel_f)
        c_c = c_c + np.dot(shift_y_tm1, self.kernel_c)

        if self.use_bias:
            c_f = np.add(c_f, self.bias_f)
            c_c = np.add(c_c, self.bias_c)

        tau_f = np.tanh(c_f)
        tau_c = np.tanh(c_c)
        c_tm2 = .5 * (c_tm1 + c_tm1 * tau_f + tau_c - tau_c * tau_f)

        u = np.dot(c_tm1, self.output_kernel)
        # print u
        x_tm2 = plant.np_step(x_tm1, u)
        return [x_tm2, c_tm2]

    def mat_save(self):
        # dirname = os.path.dirname(__file__)
        dirname = os.path.join('/users/shenshen/SafeDNN/data/matlab/')
        dirname = dirname + self.plant_name + '/'
        file_name = dirname + 'unit' + \
            str(self.units) + 'step' + str(self.steps) + self.tag + '.mat'

        confi_name = dirname + self.tag + 'config.mat'

        scipy.io.savemat(file_name, dict(plant_name=self.plant_name,
                                         num_units=self.units,
                                         recurrent_kernel_f=self.recurrent_kernel_f,
                                         recurrent_kernel_c=self.recurrent_kernel_c,
                                         kernel_f=self.kernel_f,
                                         kernel_c=self.kernel_c,
                                         output_kernel=self.output_kernel))

        scipy.io.savemat(confi_name,  dict(file_name=file_name,
                                           num_units=self.units))
        print('saved' + file_name)