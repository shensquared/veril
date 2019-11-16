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
import ClosedLoop
from CustomLayers import JanetController
import SampledLyap
from util.plotFunnel import plotFunnel
from util.samples import withinLevelSet
import itertools

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

# TODO: unify verification to based on 'plant_name' instead of hard-coded


def verifyVDP(max_deg=3, method='SGD'):
    vdp = ClosedLoop.VanderPol()
    sym_x = vdp.sym_x
    model = None
    train_x = np.load('../data/VDP/stableSamples.npy')
    # V = vdp.knownROA()
    # train_x, train_y = withinLevelSet(sym_x, V)
    vdp.set_features(max_deg)
    [phi, dphidx, f] = vdp.get_features(train_x)

    verifierOptions = Verifier.opt(vdp.num_states, vdp.degf, do_balance=False,
                                   degV=2 * max_deg, converged_tol=1e-2,
                                   max_iterations=20)
    for i in range(verifierOptions.max_iterations):
        if method is 'SGD':
            y = np.zeros(phi.shape)
            if model is None:
                model = SampledLyap.polyModel(vdp.num_states, max_deg)
            history = model.fit([phi, dphidx, f], y, epochs=15)
            if history.history['loss'][-1] >= 0:
                break
            else:
                P = SampledLyap.GetGram(model)
        else:
            P = Verifier.LPCandidate(phi, dphidx, f, num_samples=None)
        V0 = vdp.sym_phi.T@P@vdp.sym_phi
        V = Verifier.levelsetMethod(
            vdp.sym_x, V0, vdp.sym_f, verifierOptions)
        plotFunnel(vdp.sym_x, V)
    return [model, V]


def verifyClosedLoop(max_deg=2):
    CL, model_file_name = ClosedLoop.get_NNorCL(**options)
    augedSys = ClosedLoop.TanhPolyCL(CL, model_file_name)
    augedSys.set_features(max_deg)
    samples = augedSys.sampleInitialStatesInclduingTanh(100)
    [phi, dphidx, f] = augedSys.get_features(samples)

    y = np.zeros(phi.shape)
    nx = augedSys.num_states
    degf = augedSys.degf
    model = SampledLyap.polyModel(nx, max_deg)
    history = model.fit([phi, dphidx, f], y, epochs=15)
    verifierOptions = Verifier.opt(nx, degf, do_balance=False, degV=2 *
                                   max_deg, converged_tol=1e-2,
                                   max_iterations=20)
    for i in range(verifierOptions.max_iterations):
        if history.history['loss'][-1] >= 0:
            break
        else:
            P = SampledLyap.GetGram(model)
            V0 = augedSys.sym_phi.T@P@augedSys.sym_phi
            V = Verifier.levelsetMethod(
                augedSys.sym_x, V0, augedSys.sym_f, verifierOptions)


def SGDLevelSetGramCandidate(model,V, max_deg=3):
    vdp = ClosedLoop.VanderPol()
    sym_x = vdp.sym_x
    # train_x = np.load('../data/VDP/stableSamples.npy')
    train_x = withinLevelSet(list(V.GetVariables()), V)[0]
    train_y = np.ones((train_x.shape[0], 1))
    # y=[y,y]
    vdp.set_features(max_deg)
    [phi, dphidx, f] = vdp.get_features(train_x)
    P = SampledLyap.GetGram(model)
    symV = vdp.sym_phi.T@P@vdp.sym_phi
    symVdot = symV.Jacobian(sym_x) @ vdp.sym_f
    # deg = (- 1 + vdp.degf + Polynomial(symVdot, sym_x).TotalDegree()) / 2
    deg =5
    y = list(itertools.combinations_with_replacement(
        np.append(1, sym_x), 8))
    symLphi = np.stack([np.prod(j) for j in y])[1:]

    symxxd = (sym_x.T@sym_x)**np.floor(deg)
    x = train_x.T
    V = np.zeros((x.shape[-1], 1))
    Vdot = np.zeros((x.shape[-1], 1))
    xxd = np.zeros((x.shape[-1], 1))
    Lphi = np.zeros((x.shape[-1], symLphi.shape[0]))
    for i in range(x.shape[-1]):
        env = dict(zip(sym_x, x[:, i]))
        V[i, :] = symV.Evaluate(env)
        Vdot[i, :] = symVdot.Evaluate(env)
        xxd[i, :]  = symxxd.Evaluate(env)
        Lphi[i, :] = [i.Evaluate(env) for i in symLphi]

    verifyModel = SampledLyap.GramDecompModelForLevelsetPoly(vdp.num_states, max_deg)
    history = verifyModel.fit([phi, xxd, V, Vdot, Lphi], train_y, epochs=200)
    return verifyModel
    # plotFunnel(vdp.sym_x, V)
    # return V

[model,V]= verifyVDP(method='SGD')
# verifyModel=SGDLevelSetGramCandidate(model,V)
# [gram, rho, L]=SampledLyap.GetLevelsetGram(verifyModel)
# print(rho)
# verifyClosedLoop()
# train(**options)
