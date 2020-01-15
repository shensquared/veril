import os
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.layers.advanced_activations import ReLU, Softmax
from keras import backend as K

# from keras.callbacks import TensorBoard
from keras import regularizers, initializers
from keras.utils import CustomObjectScope

from math import factorial as fact
from veril.custom_layers import *
import numpy as np
'''
A note about the DOT layer: if input is 1: (None, a) and 2: (None, a) then no
need to do transpose, direct Dot()(1,2) and the output works correctly with
shape (None, 1).

If input is 1: (None, a, b) and 2: (None, b)

If debug, use kernel_initializer= initializers.Ones()
If regularizing, use kernel_regularizer= regularizers.l2(0.))

    # if write_log:
    #     logs_dir = "/Users/shenshen/veril/data/"
    #     tensorboard = TensorBoard(log_dir=logs_dir, histogram_freq=0,
    #                               write_graph=True, write_images=True)
    #     callbacks.append(tensorboard)
'''


def max_pred(y_true, y_pred):
    return K.max(y_pred)


def mean_pred(y_true, y_pred):
    return y_pred


def neg_percent(y_true, y_pred):
    return K.cast(K.equal(y_true, K.sign(y_pred)), K.floatx())


def mean_pos(y_true, y_pred):
    return K.maximum(y_pred, 0.)


def max_min(y_true, y_pred):
    # maximize the V value among all the samples such that Vdot is negative
    # return K.maximum((K.minimum(y_pred,5e-2)), 0.)
    # vdotsign = K.sign(y_pred[-1])
    # signed_V = vdotsign * y_pred[0]
    return -K.min(y_pred)


def sign_loss(y_true, y_pred):
    return K.sign(y_pred)


def flipped_relu(y_true, y_pred):
    return - K.maximum(y_pred, 0.)


def rho_reg(weight_matrix):
    return 0.001 * K.abs(K.sum(weight_matrix))

# def test_idx(y_true,y_pred):
    # return K.gather(y_pred,1)

# def hinge_and_max(y_true, y_pred):
#     return K.max(y_pred) + 10 * K.mean(K.maximum(1. - y_true * y_pred, 0.),
#                                        axis=-1)


# def guided_MSE(y_true, y_pred):
#     # want slighty greater than one
#     if K.sign(y_pred - y_true) is 1:
#         return 0.1 * K.square(y_pred - y_true)
#     else:
#         return K.square(y_pred - y_true)

# def max_and_sign(y_true, y_pred):
#     # return K.cast(K.equal(y_true, K.sign(y_pred)), K.floatx()) +
#     # .001*K.max(y_pred)
#     return K.sign(y_pred)


def model_V(system):
    sys_dim = system.num_states
    degFeatures = system.degFeatures
    remove_one = system.remove_one

    monomial_dim = get_dim(sys_dim, degFeatures, remove_one)
    phi = Input(shape=(monomial_dim,), name='phi')
    layers = [
        Dense(monomial_dim, use_bias=False),
        # Dense((monomial_dim * 2), use_bias=False),
    ]
    gram_factor = Sequential(layers, name='gram_factorization')
    phiL = gram_factor(phi)  # (None, monomial_dim)
    V = Dot(1, name='V')([phiL, phiL])  # (None,1)

    is_cl_sys = system.loop_closed
    # depending on if the loop is closed, only eta term would differ
    if not is_cl_sys:
        B = system.ctrl_B.T
        u_dim = system.num_inputs
        degU = system.degU
        ubasis_dim = get_dim(sys_dim, degU, remove_one)
        ubasis = Input(shape=(ubasis_dim,), name='ubasis')
        u = Dense(u_dim, use_bias=False, name='u')(ubasis)
        g = Input(shape=(sys_dim,), name='open_loop_dynamics')
        Bu = DotKernel(B, name='Bu')(u)
        f_cl = Add(name='closed_loop_dynamics')([g, Bu])
        dphidx = Input(shape=(monomial_dim, sys_dim), name='dphidx')
        eta = Dot(-1, name='eta')([dphidx, f_cl])
        features = [g, phi, dphidx, ubasis]

    else:
        eta = Input(shape=(monomial_dim,), name='eta')
        features = [phi, eta]

    # Vdot
    etaL = gram_factor(eta)  # (None, monomial_dim)
    Vdot = Dot(1, name='Vdot')([phiL, etaL])  # (None,1)

    # Vsqured = Power(2, name='Vsqured')(V)  # (None,1)

    # Vdot_sign = Sign(name='Vdot_sign')(Vdot)
    # V_signed = Dot(1, name='Vsigned')([V, Vdot_sign])
    # min_pos = Min_Positive(name='V-flipped')(rate)
    # diff = Subtract()([rate, min_pos])
    # rectified_V = ReLu(name = 'rectified_V')(V)
    rate = Divide(name='rate')([Vdot, V])
    model = Model(inputs=features, outputs=rate)
    model.compile(loss=mean_pos, metrics=[max_pred, mean_pred, neg_percent],
                  optimizer='adam')
    print(model.summary())
    return model


def get_model_weights(model):
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    u_weights, gram_weights = [], []
    for name, weight in zip(names, weights):
        if name.startswith('u'):
            u_weights.append(weight)
        else:
            gram_weights.append(weight)

    if len(gram_weights) == 1:
        L = gram_weights[0]
    else:
        L = np.linalg.multi_dot(gram_weights)

    if len(u_weights)==0:
        return L@L.T
    elif len(u_weights)==1:
        return L@L.T, u_weights[0]
    else:
        return L@L.T, np.linalg.multi_dot(u_weights)


def get_V_model(sys_name, tag):
    model_dir = os.path.dirname(__file__) + '/../data/' + sys_name
    file_name = model_dir + '/V_model' + tag + '.h5'

    with CustomObjectScope({'Divide': Divide, 'DotKernel': DotKernel,
                            'max_pred': max_pred, 'mean_pred': mean_pred,
                            'neg_percent': neg_percent, 'mean_pos': mean_pos}):
        model = load_model(file_name)
    print(model.summary())
    return model


# def model_UV(plant):
#     # assuming control affine
#     B = plant.ctrl_B.T
#     u_dim = plant.num_inputs
#     sys_dim = plant.num_states
#     degFeatures = plant.degFeatures
#     degU = plant.degU
#     remove_one = plant.remove_one

#     monomial_dim = get_dim(sys_dim, degFeatures, remove_one)
#     ubasis_dim = get_dim(sys_dim, degU, remove_one)

#     phi = Input(shape=(monomial_dim,), name='phi')
#     layers = [
#         Dense(monomial_dim, use_bias=False),
#         # Dense((monomial_dim * 2), use_bias=False),
#         # Dense(monomial_dim, use_bias=False),
#     ]
#     gram_factor = Sequential(layers, name='gram_factorization')
#     phiL = gram_factor(phi)  # (None, monomial_dim)
#     V = Dot(1, name='V')([phiL, phiL])  # (None,1)

#     # # need to avoid 0 in the denominator (by adding a strictly positive scalar
#     # # to V)
#     ubasis = Input(shape=(ubasis_dim,), name='ubasis')
#     u = Dense(u_dim, use_bias=False, name='u')(ubasis)
#     g = Input(shape=(sys_dim,), name='open_loop_dynamics')

#     Bu = DotKernel(B, name='Bu')(u)
#     f_cl = Add(name='closed_loop_dynamics')([g, Bu])

#     dphidx = Input(shape=(monomial_dim, sys_dim), name='dphidx')
#     eta = Dot(-1, name='eta')([dphidx, f_cl])
#     etaL = gram_factor(eta)  # (None, monomial_dim)
#     Vdot = Dot(1, name='Vdot')([phiL, etaL])  # (None,1)
#     rate = Divide(name='ratio')([Vdot, V])

#     features = [g, phi, dphidx, ubasis]
#     model = Model(inputs=features, outputs=rate)
#     model.compile(loss=max_pred, metrics=[mean_pred, max_pred, neg_percent],
#                   optimizer='adam')
#     print(model.summary())
#     return model

# def gram_decomp_model_for_levelsetpoly(sys_dim, sigma_deg, psi_deg):
#     psi_dim = get_dim(sys_dim, psi_deg)
#     psi = Input(shape=(psi_dim,), name='psi')
#     layers = [
#         Dense(psi_dim, use_bias=False),
#         # Dense(math.floor(psi_dim / 2), use_bias=False),
#         # Dense(10, use_bias=False),
#         # Dense(4, use_bias=False),
#     ]
#     layers = layers + [TransLayer(i) for i in layers[::-1]]
#     psiLL = Sequential(layers, name='Gram')(psi)  # (None,
#     # monomial_dim)
#     candidateSOS = Dot(1)([psiLL, psi])  # (None,1)

#     xxd = Input(shape=(1,), name='xxd')
#     V = Input(shape=(1,), name='V')
#     Vdot = Input(shape=(1,), name='Vdot')

#     xxdV = Dot(-1)([xxd, V])

#     sigma_dim = get_dim(sys_dim + sigma_deg)
#     sigma = Input(shape=(sigma_dim,), name='sigma')

#     multiplierLayers = [
#         # Dense(sigma_dim, use_bias=False),
#         # Dense(4, use_bias=False),
#         Dense(1, use_bias=False),
#     ]
#     L1 = Sequential(multiplierLayers, name='multiplier')(sigma)
#     L1Vdot = Dot(-1, name='L1Vdot')([L1, Vdot])

#     rholayer = [Dense(1, use_bias=False, kernel_regularizer=rho_reg)]
#     # kernel_regularizer=rho_reg
#     rholayer = rholayer + [TransLayer(i) for i in rholayer[::-1]]

#     # residual = sos - (xxdV-xxdrho+L1Vdot)
#     # if use xxd(V-rho)
#     # xxdrho = Sequential(rholayer, name='rho')(xxd)
#     # vminusrho = Subtract()([xxdV, xxdrho])
#     # if use xxd(rho*V - 1)
#     xxdVrho = Sequential(rholayer, name='rho')(xxdV)
#     vminusrho = Subtract()([xxdVrho, xxd])
#     vminusrhoplusvdot = Add()([vminusrho, L1Vdot])
#     ratio = Divide()([vminusrhoplusvdot, candidateSOS])
#     # outputs = Dot(-1)([ratio, xxdrho])
#     model = Model(inputs=[V, Vdot, xxd, psi, sigma], outputs=ratio)
#     model.compile(loss='mse', metrics=[mean_pred, 'mse'],
#                   optimizer='adam')
#     print(model.summary())
#     return model


# def linear_model_for_V(sys_dim, A):
#     x = Input(shape=(sys_dim,))  # (None, sys_dim)
#     layers = [
#         Dense(5, input_shape=(sys_dim,), use_bias=False,
#               kernel_regularizer=regularizers.l2(0.)),
#         Dense(2, use_bias=False, kernel_regularizer=regularizers.l2(0.)),
#         Dense(2, use_bias=False, kernel_regularizer=regularizers.l2(0.))
#     ]
#     layers = layers + [TransLayer(i) for i in layers[::-1]]
#     xLL = Sequential(layers)(x)  # (None, sys_dim)
#     # need to avoid 0 in the denominator (by adding a strictly positive scalar
#     # to V)
#     V = Dot(1)([xLL, x])  # (None, 1)
#     xLLA = DotKernel(A)(xLL)  # A: (sys_dim,sys_dim), xLLA: (None, sys_dim)
#     Vdot = Dot(1)([xLLA, x])  # Vdot: (None,1)
#     rate = Divide()([Vdot, V])
#     model = Model(inputs=x, outputs=rate)
#     model.compile(loss=max_pred, optimizer='adam')
#     print(model.summary())
#     return model


def get_dim(num_var, deg, remove_one):
    if remove_one:
        return fact(num_var + deg) // fact(num_var) // fact(deg) - 1
    else:
        return fact(num_var + deg) // fact(num_var) // fact(deg)
# def get_gram_trans_for_levelset_poly(model):
#     names = [weight.name for layer in model.layers for weight in layer.weights]
#     weights = model.get_weights()
#     gram_weights = []
#     rho_weights = []
#     L_weights = []
#     for name, weight in zip(names, weights):
#         if 'Gram' in name:
#             gram_weights = gram_weights + [weight]
#         elif 'rho' in name:
#             rho_weights = rho_weights + [weight]
#         elif 'multiplier' in name:
#             L_weights = L_weights + [weight]
#         else:
#             print('should not have other weights')
#     if len(gram_weights) == 1:
#         g = gram_weights[0]
#     else:
#         g = np.linalg.multi_dot(gram_weights)
#     gram = g@g.T
#     print('cond # of the candidate gram: %s' %np.linalg.cond(gram))
#     rho = rho_weights[0]**2
#     if len(L_weights) == 1:
#         L = L_weights[0]
#     else:
#         L = np.linalg.multi_dot(L_weights)
#     return [gram, g, rho, L]
