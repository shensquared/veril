from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras import backend as K
# from keras.callbacks import TensorBoard
from CustomLayers import *
import keras
from keras.utils import CustomObjectScope

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


def negativity(y_true, y_pred):
    return K.max(y_pred)


def get_data(d=30, num_grid=100):
    x1 = np.linspace(-d, d, num_grid)
    x2 = np.linspace(-d, d, num_grid)
    x1 = x1[np.nonzero(x1)]
    x2 = x2[np.nonzero(x2)]
    x1, x2 = np.meshgrid(x1, x2)
    x1, x2 = x1.ravel(), x2.ravel()
    return [np.array([x1, x2]).T, np.zeros(x1.shape)]


def linear_model(sys_dim, A):
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
    xLLA = DotKernel(A)(xLL) # A: (sys_dim,sys_dim), xLLA: (None, sys_dim)
    Vdot = Dot(1)([xLLA, x])  # Vdot: (None,1)
    rate = Divide()([Vdot, V])

    model = Model(inputs=x, outputs=rate)
    model.compile(loss=negativity, optimizer='adam')
    print(model.summary())
    return model


def linear_train():
    callbacks = []
    x, y = get_data()
    A = np.array([[-.1, 0], [1, -2]])
    model = linear_model(2, A)
    print(model.predict(x))
    history = model.fit(x, y, epochs=15, verbose=True, callbacks=callbacks)
    # getting the weights and verify the eigs
    weights = [K.eval(i) for i in model.weights]
    weights = np.linalg.multi_dot(weights)
    P = weights@weights.T
    print('eig of orignal PA+A\'P  %s' % (np.linalg.eig(P@A + A.T@P)[0]))
    model_file_name = '/Users/shenshen/Veril/data/Kernel/lyap_model.h5'
    model.save(model_file_name)
    del model
    print("Saved model" + model_file_name + "to disk")
    return P


def poly_model(sys_dim, max_deg=4):
    # constant = Input(shape=(1,))
    x = Input(shape=(sys_dim,))  # x: (None, sys_dim)
    phi = Polynomials(max_deg)(x)  # phi: (None, monomial_dim)
    # phi = Concatenate()([constant,phi])

    layers = [
        Dense(10, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.)),
        Dense(5, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.)),
        Dense(10, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.))
    ]
    layers = layers + [TransLayer(i) for i in layers[::-1]]
    phiLL = Sequential(layers)(phi)  # phiLL: (None, monomial_dim)

    # # need to avoid 0 in the denominator (by adding a strictly positive scalar
    # # to V)
    V = Dot(1)([phiLL, phi])  # V: (None,1)
    dphidx = DiffPoly(max_deg)(x)  # dphidx: (None, monomial_dim, sys_dm)
    phiLLdphidx = Dot(1)([phiLL, dphidx])  # (None, sys_dim)
    fx = Lambda(VDP, Poly_dynamics_shape)(x) # (None, sys_dim)
    Vdot = Dot(1)([phiLLdphidx, fx])  # (None, 1)

    rate = Divide()([Vdot, V])
    model = Model(inputs=x, outputs=rate)
    model.compile(loss=negativity, optimizer='adam')
    print(model.summary())
    return model


def VDP(x):
    x1 = (K.dot(x, K.constant([1, 0], shape=(2, 1))))
    x2 = (K.dot(x, K.constant([0, 1], shape=(2, 1))))
    dx = (K.concatenate([-x2, -(1 - x1**2) * x2 + x1]))
    return dx


def Poly_dynamics_shape(input_shapes):
    return input_shapes


def poly_train():
    callbacks = []
    sys_dim =2
    x, y = get_data(d=1)
    model = poly_model(sys_dim, max_deg = 3)
    # print(model.predict(x))
    history = model.fit(x, y, epochs=150, verbose=True,
                        callbacks=callbacks)
    # weights = [K.eval(i) for i in model.weights]
    # print(weights)
    # weights = np.linalg.multi_dot(weights)
    # P = weights@weights.T
    # print(np.linalg.eig(P)[0])
    # print(np.linalg.eig(P@A + A.T@P)[0])
    # model_file_name = '/Users/shenshen/Veril/data/Kernel/lyap_model.h5'
    # model.save(model_file_name)
    # return P
poly_train()