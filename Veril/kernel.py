from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras import backend as K
# from keras.callbacks import TensorBoard
from CustomLayers import *
import keras
from keras.utils import CustomObjectScope
from Verifier import *
from util.plotFunnel import *
from util.samples import *
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
'''
A note about the DOT layer: if input is 1: (None, a) and 2: (None, a) then no
need to do transpose, direct Dot()(1,2) and the output works correctly with
shape (None, 1).

If input is 1: (None, a, b) and 2: (None, b)

If debug, use kernel_initializer= keras.initializers.Ones()

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
    x = Input(shape=(sys_dim,))  # x: (None, sys_dim)
    layers = [
        Dense(5, input_shape=(sys_dim,), use_bias=False,
              kernel_regularizer=keras.regularizers.l2(0.)),
        Dense(2, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.)),
        Dense(2, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.))
    ]
    layers = layers + [TransLayer(i) for i in layers[::-1]]

    xLL = Sequential(layers)(x)  # xLL shape (None, sys_dim)
    # need to avoid 0 in the denominator (by adding a strictly positive scalar
    # to V)
    V = Dot(1)([xLL, x])  # V: (None, 1)
    xLLA = DotKernel(A)(xLL)  # A: (sys_dim,sys_dim), xLLA: (None, sys_dim)
    Vdot = Dot(1)([xLLA, x])  # Vdot: (None,1)
    rate = Divide()([Vdot, V])

    model = Model(inputs=x, outputs=rate)
    model.compile(loss=max_negativity, optimizer='adam')
    print(model.summary())
    return model


def linearTrain():
    callbacks = []
    x, y = get_data()
    A = np.array([[-.1, 0], [1, -2]])
    model = linearModel(2, A)
    print(model.predict(x))
    history = model.fit(x, y, epochs=15, verbose=True, callbacks=callbacks)
    # getting the weights and verify the eigs
    weights = [K.eval(i) for i in model.weights]
    weights = np.linalg.multi_dot(weights)
    P = weights@weights.T
    print('eig of orignal PA+A\'P  %s' % (np.linalg.eig(P@A + A.T@P)[0]))
    model_file_name = '../data/Kernel/lyap_model.h5'
    model.save(model_file_name)
    del model
    print("Saved model" + model_file_name + " to disk")
    return P


def VDP(x):
    x1 = (K.dot(x, K.constant([1, 0], shape=(2, 1))))
    x2 = (K.dot(x, K.constant([0, 1], shape=(2, 1))))
    dx = (K.concatenate([-x2, -(1 - x1**2) * x2 + x1]))
    return dx


def PolyDynamicsOutputShape(input_shapes):
    return input_shapes


def polyModel(sys_dim, max_deg):
    # constant = Input(shape=(1,))
    x = Input(shape=(sys_dim,))  # x: (None, sys_dim)
    phi = Polynomials(sys_dim, max_deg)(x)  # phi: (None, monomial_dim)
    # phi = Concatenate()([constant,phi])
    # kernel_regularizer=keras.regularizers.l2(0.))
    layers = [
        Dense(20, use_bias=False),
        Dense(8, use_bias=False),
        Dense(10, use_bias=False),
        Dense(8, use_bias=False),
    ]
    layers = layers + [TransLayer(i) for i in layers[::-1]]
    phiLL = Sequential(layers)(phi)  # phiLL: (None, monomial_dim)

    # # need to avoid 0 in the denominator (by adding a strictly positive scalar
    # # to V)
    V = Dot(1)([phiLL, phi])  # V: (None,1)
    dphidx = DiffPoly(sys_dim, max_deg)(x)  # (None, monomial_dim,sys_dm)
    phiLLdphidx = Dot(1)([phiLL, dphidx])  # (None, sys_dim)
    fx = Lambda(VDP, PolyDynamicsOutputShape)(x)  # (None, sys_dim)
    Vdot = Dot(1)([phiLLdphidx, fx])  # (None, 1)

    rate = Divide()([Vdot, V])
    model = Model(inputs=x, outputs=rate)
    model.compile(loss=max_negativity, optimizer='adam')
    print(model.summary())
    return model


def polyTrain(nx, max_deg,  x, V=None, model=None):
    callbacks = []
    if V is None:
        train_x, train_y = get_data(d=1, num_grid=200)
    else:
        train_x, train_y = withinLevelSet(x, V)

    if model is None:
        model = polyModel(nx, max_deg)
    # prog = MathematicalProgram()
    # x = prog.NewIndeterminates(nx, "x")
    # f = -np.array([x[1], -x[0] - x[1] * (x[0]**2 - 1)])
    # y = list(itertools.combinations_with_replacement(np.append(1, x), max_deg))
    # phi = np.stack([np.prod(j) for j in y])[1:]
    # train_x = np.array([[2.1,3.7]])
    # train_y = train_x
    # print('model pridcted')
    # print(model.predict(train_x))
    # env = dict(zip(x, train_x.T))
    # print([i.Substitute(env) for i in phi])

    history = model.fit(train_x, train_y, batch_size=32,
                        shuffle=True, epochs=15,
                        verbose=True,
                        callbacks=callbacks)
    weights = [K.eval(i) for i in model.weights]
    weights = np.linalg.multi_dot(weights)
    P = weights@weights.T
    # V0 = phi.T@P@phi
    # Vdot = V0.Jacobian(x) @ f
    # print(Vdot.Substitute(env))
    model_file_name = '../data/Kernel/polyModel.h5'
    model.save(model_file_name)
    print("Saved model" + model_file_name + " to disk")
    return P, model, history


def run():
    nx = 2
    degf = 3
    max_deg = 3
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(nx, "x")
    f = -np.array([x[1], -x[0] - x[1] * (x[0]**2 - 1)])
    y = list(itertools.combinations_with_replacement(np.append(1, x), max_deg))
    phi = np.stack([np.prod(j) for j in y])[1:]

    options = opt(nx, degf, do_balance=False, degV=2 * max_deg,
                  converged_tol=1e-2, max_iterations=20)

    x1 = x[0]
    x2 = x[1]
    V = (1.8027e-06) + (0.28557) * x1**2 + (0.0085754) * x1**4 + \
        (0.18442) * x2**2 + (0.016538) * x2**4 + \
        (-0.34562) * x2 * x1 + (0.064721) * x2 * x1**3 + \
        (0.10556) * x2**2 * x1**2 + (-0.060367) * x2**3 * x1
    V=5*V

    # plotFunnel(x, V)
    model = None
    for i in range(options.max_iterations):
        P, model, history = polyTrain(nx, max_deg, x, V, model)
        if history.history['loss'][-1] >= 0:
            break
        else:
            file_name = '../data/Kernel/poly_P.npy'
            np.save(file_name, P)
            # P = np.load('/Users/shenshen/Veril/data/Kernel/poly_P_new.npy')
            V0 = phi.T@P@phi
            V = levelsetMethod(x, V0, f, options)
            # V = bilinear(x, V0, f, None, None, options)
            plotFunnel(x, V)
    return V

V = run()
print(V)
