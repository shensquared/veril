from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras import backend as K
# from keras.callbacks import TensorBoard
from keras import regularizers, initializers
from keras.utils import CustomObjectScope

import math
from CustomLayers import DotKernel, Divide, TransLayer
import numpy as np
'''
A note about the DOT layer: if input is 1: (None, a) and 2: (None, a) then no
need to do transpose, direct Dot()(1,2) and the output works correctly with
shape (None, 1).

If input is 1: (None, a, b) and 2: (None, b)

If debug, use kernel_initializer= initializers.Ones()
If regularizing, use kernel_regularizer= regularizers.l2(0.))

    # if write_log:
    #     logs_dir = "/Users/shenshen/Veril/data/lyapunov"
    #     tensorboard = TensorBoard(log_dir=logs_dir, histogram_freq=0,
    #                               write_graph=True, write_images=True)
    #     callbacks.append(tensorboard)
'''


def max_negativity(y_true, y_pred):
    return K.max(y_pred)


def linearModel(sys_dim, A):
    x = Input(shape=(sys_dim,))  # (None, sys_dim)
    layers = [
        Dense(5, input_shape=(sys_dim,), use_bias=False,
              kernel_regularizer=regularizers.l2(0.)),
        Dense(2, use_bias=False, kernel_regularizer=regularizers.l2(0.)),
        Dense(2, use_bias=False, kernel_regularizer=regularizers.l2(0.))
    ]
    layers = layers + [TransLayer(i) for i in layers[::-1]]
    xLL = Sequential(layers)(x)  # (None, sys_dim)
    # need to avoid 0 in the denominator (by adding a strictly positive scalar
    # to V)
    V = Dot(1)([xLL, x])  # (None, 1)
    xLLA = DotKernel(A)(xLL)  # A: (sys_dim,sys_dim), xLLA: (None, sys_dim)
    Vdot = Dot(1)([xLLA, x])  # Vdot: (None,1)
    rate = Divide()([Vdot, V])
    model = Model(inputs=x, outputs=rate)
    model.compile(loss=max_negativity, optimizer='adam')
    print(model.summary())
    return model


# def linearTrain():
#     callbacks = []
#     x, y = get_data()
#     A = np.array([[-.1, 0], [1, -2]])
#     model = linearModel(2, A)
#     print(model.predict(x))
#     history = model.fit(x, y, epochs=15, verbose=True, callbacks=callbacks)
#     # getting the weights and verify the eigs
#     L = np.linalg.multi_dot(model.get_weights())
#     P = L@L.T
#     print('eig of orignal PA+A\'P  %s' % (np.linalg.eig(P@A + A.T@P)[0]))
#     model_file_name = '../data/Kernel/lyap_model.h5'
#     model.save(model_file_name)
#     del model
#     print("Saved model" + model_file_name + " to disk")
#     return P

# TODO lambda f (two args in)
def polyModel(sys_dim, max_deg):
    f = lambda x: math.factorial(x)
    # -1 since V doesn't have a constant monomial
    monomial_dim = f(sys_dim + max_deg) // f(max_deg) // f(sys_dim)
    phi = Input(shape=(monomial_dim,))
    layers = [
        # Dense(monomial_dim, use_bias=False),
        # Dense(math.floor(monomial_dim / 2), use_bias=False),
        Dense(monomial_dim, use_bias=False),
    ]
    layers = layers + [TransLayer(i) for i in layers[::-1]]
    phiLL = Sequential(layers)(phi)  # (None, monomial_dim)

    # # need to avoid 0 in the denominator (by adding a strictly positive scalar
    # # to V)
    V = Dot(1)([phiLL, phi])  # (None,1)
    dphidx = Input(shape=(monomial_dim, sys_dim,))
    phiLLdphidx = Dot(1)([phiLL, dphidx])  # (None, sys_dim)
    fx = Input(shape=(sys_dim,))
    Vdot = Dot(1)([phiLLdphidx, fx])  # (None, 1)
    rate = Divide()([Vdot, V])
    model = Model(inputs=[phi, dphidx, fx], outputs=rate)
    model.compile(loss=max_negativity, optimizer='adam')
    print(model.summary())
    return model


# def polyTrain(nx, max_deg, x, V=None, model=None):
#     if model is None:
#         model = polyModel(nx, max_deg)
#     history = model.fit(train_x, train_y, shuffle=True, epochs=15,
#        verbose=True)
#     P = GetGram(model)
#     model_file_name = '../data/Kernel/polyModel.h5'
#     model.save(model_file_name)
#     print("Saved model" + model_file_name + " to disk")
#     return P, model, history

def GetGram(model):
    weights = model.get_weights()
    if len(weights) == 1:
        L = weights[0]
    else:
        L = np.linalg.multi_dot(weights)
    return L@L.T


def GramDecompModelForLevelsetPoly(sys_dim, sigma_deg, psi_deg):
    f = lambda x: math.factorial(x)
    # -1 since V doesn't have a constant monomial
    psi_dim = f(sys_dim + psi_deg) // f(psi_deg) // f(sys_dim)
    psi = Input(shape=(psi_dim,), name='psi')
    layers = [
        Dense(psi_dim, use_bias=False),
        # Dense(math.floor(psi_dim / 2), use_bias=False),
        # Dense(10, use_bias=False),
        # Dense(4, use_bias=False),
    ]
    layers = layers + [TransLayer(i) for i in layers[::-1]]
    psiLL = Sequential(layers, name='Gram')(psi)  # (None,
    # monomial_dim)
    candidateSOS = Dot(1)([psiLL, psi])  # (None,1)

    xxd = Input(shape=(1,), name='xxd')
    V = Input(shape=(1,), name='V')
    Vdot = Input(shape=(1,), name='Vdot')

    xxdV = Dot(-1)([xxd, V])

    sigma_dim = f(sys_dim + sigma_deg) // f(sigma_deg) // f(sys_dim)
    sigma = Input(shape=(sigma_dim,), name='sigma')

    multiplierLayers = [
        # Dense(sigma_dim, use_bias=False),
        # Dense(4, use_bias=False),
        Dense(1, use_bias=False),
    ]
    L1 = Sequential(multiplierLayers, name='multiplier')(sigma)
    L1Vdot = Dot(-1, name='L1Vdot')([L1, Vdot])

    rholayer = [Dense(1, use_bias=False, kernel_regularizer=rho_reg)]
    # kernel_regularizer=rho_reg
    rholayer = rholayer + [TransLayer(i) for i in rholayer[::-1]]
    xxdVrho = Sequential(rholayer, name='rho')(xxdV)
    # xxdrho = xxd
    # residual = sos - (xxdVrho-xxdrho+L1Vdot)
    vminusrho = Subtract()([xxdVrho, xxd])
    vminusrhoplusvdot = Add()([vminusrho, L1Vdot])
    ratio = Divide()([vminusrhoplusvdot, candidateSOS])
    # outputs = Dot(-1)([ratio, xxdrho])
    model = Model(inputs=[V, Vdot, xxd, psi, sigma], outputs=ratio)
    model.compile(loss='mse', metrics=[mean_pred, 'mse'],
                  optimizer='adam')
    print(model.summary())
    return model


def rho_reg(weight_matrix):
    return 0.001 * K.abs(K.sum(weight_matrix))


def guidedMSE(y_true, y_pred):
    # want slighty greater than one
    if K.sign(y_pred - y_true) is 1:
        return 0.1 * K.square(y_pred - y_true)
    else:
        return K.square(y_pred - y_true)


def mean_pred(y_true, y_pred):
    return y_pred


def GetLevelsetGram(model):
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    gram_weights = []
    rho_weights = []
    L_weights = []
    for name, weight in zip(names, weights):
        if 'Gram' in name:
            gram_weights = gram_weights + [weight]
        elif 'rho' in name:
            rho_weights = rho_weights + [weight]
        elif 'multiplier' in name:
            L_weights = L_weights + [weight]
        else:
            print('should not have other weights')
    if len(gram_weights) == 1:
        g = gram_weights[0]
    else:
        g = np.linalg.multi_dot(gram_weights)
    gram = g@g.T
    rho = rho_weights[0]**2
    L = np.linalg.multi_dot(L_weights)
    return [gram, g, rho, L]
