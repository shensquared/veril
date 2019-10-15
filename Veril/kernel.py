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
    x = Input(shape=(sys_dim,)) # x shape (None, sys_dim)
    layers = [
        Dense(5, input_shape=(sys_dim,), use_bias=False,
              kernel_regularizer=keras.regularizers.l2(0.)),
        Dense(2, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.)),
        Dense(2, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.))
    ]
    layers = layers + [TransLayer(i) for i in layers[::-1]]

    xLL = Sequential(layers)(x)# xLL shape (None, sys_dim)
    # need to avoid 0 in the denominator (by adding a strictly positive scalar
    # to V)
    V = Dot(1)([xLL, x]) # V shape (None, 1)
    xLLA = DotKernel(A)(xLL) #A shape (sys_dim,sys_dim), xLLA: (None, sys_dim)
    Vdot = Dot(1)([xLLA, x]) #Vdot (None,1)
    rate = Divide()([Vdot, V])

    model = Model(inputs=x, outputs=rate)
    model.compile(loss=negativity, optimizer='adam')
    print(model.summary())
    return model

def linear_train():
    callbacks = []
    # if write_log:
    #     logs_dir = "/Users/shenshen/Veril/data/lyapunov"
    #     tensorboard = TensorBoard(log_dir=logs_dir, histogram_freq=0,
    #                               write_graph=True, write_images=True)
    #     callbacks.append(tensorboard)
    x, y = get_data()
    A = np.array([[-.1, 0], [1, -2]])
    model = linear_model(2, A)
    print(model.predict(x))
    history = model.fit(x, y, epochs=15, verbose=True, callbacks=callbacks)
    weights = [K.eval(i) for i in model.weights]
    print(weights)
    weights = np.linalg.multi_dot(weights)
    P = weights@weights.T
    # print(np.linalg.eig(P)[0]
    print(np.linalg.eig(P@A + A.T@P)[0])
    model_file_name = '/Users/shenshen/Veril/data/Kernel/lyap_model.h5'
    model.save(model_file_name)
    return P
    # del model
    # print("Saved model" + model_file_name + "to disk")
    # model


def polynomial_model(sys_dim, max_deg):
    # constant = Input(shape=(1,))
    x = Input(shape=(sys_dim,))
    phi = Polynomials(max_deg)(x)
    # phi = Concatenate()([constant,phi])

    layers = [
        Dense(1, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.)),
        Dense(1, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.)),
        Dense(1, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.))
    ]
    layers = layers + [TransLayer(i) for i in layers[::-1]]
    seq_model = Sequential(layers)
    phiLL = seq_model(phi)
    # # need to avoid 0 in the denominator (by adding a strictly positive scalar
    # # to V)
    # V = Dot(1)([phiLL, phiLL])
    dphidx = DiffPoly(max_deg)(x)
    fx = Permute((1,2))(Lambda(VDP, Poly_dynamics_shape)(x))
    # dVdx = Lambda(threeDdot,threeDdot_shape)([phiLL,dphidx])
    dVdx = Dot(1)([phiLL,dphidx])
    Vdot = Dot(1)([dVdx, fx])
    # dVdx = SimpleDot(-1)([dphidx,phiLL])
    # Vdot = Lambda(threeDdot,threeDdot_shape)([dVdx,fx])

    # rate = Divide()([Vdot, dVdx])
    model = Model(inputs=x, outputs=Vdot)
    model.compile(loss=negativity, optimizer='adam')
    print(model.summary())
    return model

def VDP(x):
    x1 = (K.dot(x, K.constant([1, 0], shape=(2, 1))))
    x2 = (K.dot(x, K.constant([0, 1], shape=(2, 1))))
    dx = (K.concatenate([-x2, -(1 - x1**2) * x2 + x1]))
    return K.expand_dims(dx)

def Poly_dynamics_shape(input_shapes):
    input_shape = list(input_shapes)
    input_shape.append(1)
    output_shapes = tuple(input_shape)
    return output_shapes

def threeDdot(x):
    # TODO: write for more than 2 input list
    x1,x2=x
    return K.dot(x1,x2)

def threeDdot_shape(input_shapes):

    shape1=list(input_shapes[0])
    shape2=list(input_shapes[1])
    # print(shape1)
    # print(shape2)
    shape1.pop(1)
    shape2.pop(1)
    shape2.pop(0)
    output_shapes = shape1 + shape2
    # print(output_shapes)
    return tuple(output_shapes)

def poly_train():
    callbacks = []
    x, y = get_data(d=1.5,num_grid=2)
    # print(x.shape, y.shape)
    model = polynomial_model(2,3)
    print(model.predict(x))
    # history = model.fit(x, y, epochs=15, verbose=True,
    # callbacks=callbacks)
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
