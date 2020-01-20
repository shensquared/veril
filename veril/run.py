import time
import numpy as np
# import argparse
from veril.systems import get_system
from veril.sample_lyap import get_V

from veril.symbolic_verifier import verify_via_equality, verify_via_bilinear
from veril.sample_variety import verify_via_variety
from veril.util.plots import *

####### system setup ####### ####### ####### ####### ####### ####### #######
# sys_name = 'VanderPol'
# sys_name = 'Pendubot'
# sys_name = 'PendulumTrig'
sys_name = 'PendulumRecast'
deg_ftrs = 2
deg_u = 1
rm_one = True
system = get_system(sys_name, deg_ftrs, deg_u, rm_one)
# stableSamples = system.sample_stable_inits(d=1, num_grid=10)

####### ####### ####### ####### ####### ####### ####### ####### #######


####### model setup ####### ####### ####### ####### ####### ####### #######
# train_or_load = 'Train'
train_or_load = 'Load'
epochs = 50
V, Vdot, system, model, P, u_weights = get_V(system, train_or_load, epochs=epochs,
                                  verbose=True, validation_split=0,
                                  shuffle=True)

up = [np.array([0,-1,0])]
down =  np.array([np.array([0,1,0])])
print(system.get_v_values(up,V))
print(system.get_v_values(down,V))

plot3d(V, system, r_max=[np.pi,10], in_xo=True)
# plot3d(Vdot, system, r_max=[1.5*np.pi, 6])

# initial = np.array([np.array([0,-1,0])])
# initial = system.random_sample(100)
# print(initial)
# plot_traj(initial, system, int_horizon=10, V=V)


# direct  = system.get_v_values(initial,V=Vdot)
# indirect = check_v_dot(system, P, initial)
def check_v_dot(system, P, x):
    system.set_syms(deg_ftrs, deg_u)
    Vdot = system.sym_phi.T@P@system.sym_eta
    return system.get_v_values(x, V=Vdot)

####### ####### ####### ####### ####### ####### ####### ####### #######


####### Verification ####### ####### ####### ####### ####### ####### #######

init_root_threads = 25
# verify_via_bilinear(system)
# verify_via_equality(system, V)
verify_via_variety(system, V, init_root_threads=init_root_threads)
####### ####### ####### ####### ####### ####### ####### ####### #######
