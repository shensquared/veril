from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras import backend as K
# from keras.callbacks import TensorBoard
from CustomLayers import *
import keras
from keras.utils import CustomObjectScope
from Verifier import *
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
    xLLA = DotKernel(A)(xLL)  # A: (sys_dim,sys_dim), xLLA: (None, sys_dim)
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
    # ones = Input(shape=(1,))
    x = Input(shape=(sys_dim,))  # x: (None, sys_dim)
    phi = Polynomials(max_deg)(x)  # phi: (None, monomial_dim)
    # phi = Concatenate()([ones,phi])

    layers = [
        Dense(10, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.)),
        Dense(5, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.)),
        Dense(10, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.))
    ]
    layers = layers + [TransLayer(i) for i in layers[::-1]]
    phiLL = Sequential(layers)(phi)  # phiLL: (None, monomial_dim)
    # rho = ConstantLayer()(x)
    # # need to avoid 0 in the denominator (by adding a strictly positive scalar
    # # to V)
    V = Dot(1)([phiLL, phi])  # V: (None,1)
    dphidx = DiffPoly(max_deg)(x)  # dphidx: (None, monomial_dim, sys_dm)
    phiLLdphidx = Dot(1)([phiLL, dphidx])  # (None, sys_dim)
    fx = Lambda(VDP, Poly_dynamics_shape)(x)  # (None, sys_dim)
    Vdot = Dot(1)([phiLLdphidx, fx])  # (None, 1)

    rate = Divide()([Vdot, V])
    model = Model(inputs=x, outputs=rho)
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


def poly_train(max_deg=1, model=None, V=None):
    callbacks = []
    sys_dim = 2
    x, y = get_data(d=.5)
    if model is None:
        model = poly_model(sys_dim, max_deg=max_deg)
    # print(model.predict(x))
    history = model.fit(x, y, epochs=150, verbose=True,
                        callbacks=callbacks)
    weights = [K.eval(i) for i in model.weights]
    weights = np.linalg.multi_dot(weights)
    P = weights@weights.T
    # print(np.linalg.eig(P)[0])
    # print(np.linalg.eig(P@A + A.T@P)[0])
    model_file_name = '/Users/shenshen/Veril/data/Kernel/poly_model.h5'
    model.save(model_file_name)
    return P
nx = 2
options = opt(nx, converged_tol=1e-2, degL1=6, degL2=6, degV=4,
              max_iterations=100)
max_deg = 2
P = poly_train(max_deg=max_deg)

# np.save('/Users/shenshen/Veril/data/Kernel/poly_P.npy',P)
# P = np.load('/Users/shenshen/Veril/data/Kernel/poly_P.npy')
# prog = MathematicalProgram()
# x = prog.NewIndeterminates(nx, "x")
# f = -np.array([x[1], -x[0] - x[1] * (x[0]**2 - 1)])
# phi = np.array([sym.pow(j, i) for i in range(1, max_deg+1)  for j in x ])
# V0 = phi.T@P@phi
# V = levelsetMethod(x, V0, f, options)
