from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras import backend as K
# from keras.callbacks import TensorBoard
from keras import regularizers, initializers
from keras.utils import CustomObjectScope

import math
from CustomLayers import DotKernel, Divide, TransLayer
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
    # return K.mean(y_pred)


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


def polyModel(sys_dim, max_deg):
    f = lambda x: math.factorial(x)
    # -1 since V doesn't have a constant monomial
    monomial_dim = f(sys_dim + max_deg) // f(max_deg) // f(sys_dim) - 1
    phi = Input(shape=(monomial_dim,))
    layers = [
        Dense(monomial_dim, use_bias=False),
        Dense(math.floor(monomial_dim/2), use_bias=False),
        Dense(10, use_bias=False),
        Dense(4, use_bias=False),
    ]
    layers = layers + [TransLayer(i) for i in layers[::-1]]
    phiLL = Sequential(layers)(phi)  # phiLL: (None, monomial_dim)

    # # need to avoid 0 in the denominator (by adding a strictly positive scalar
    # # to V)
    V = Dot(1)([phiLL, phi])  # V: (None,1)
    dphidx = Input(shape=(monomial_dim, sys_dim,))
    phiLLdphidx = Dot(1)([phiLL, dphidx])  # (None, sys_dim)
    fx = Input(shape=(sys_dim,))
    Vdot = Dot(1)([phiLLdphidx, fx])  # (None, 1)
    rate = Divide()([Vdot, V])
    model = Model(inputs=[phi, dphidx, fx], outputs=rate)
    model.compile(loss=max_negativity, optimizer='adam')
    print(model.summary())
    return model


def polyTrain(nx, max_deg, x, V=None, model=None):
    if model is None:
        model = polyModel(nx, max_deg)
    history = model.fit(train_x, train_y, batch_size=32,
                        shuffle=True, epochs=15,
                        verbose=True)
    L = np.linalg.multi_dot(model.get_weights())
    P = L@L.T
    model_file_name = '../data/Kernel/polyModel.h5'
    model.save(model_file_name)
    print("Saved model" + model_file_name + " to disk")
    return P, model, history
