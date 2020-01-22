# import time, argparse
import numpy as np
from veril.systems import get_system
from veril.sample_lyap import get_V
from veril.symbolic_verifier import verify_via_equality, verify_via_bilinear
from veril.sample_variety import verify_via_variety
from veril.util.plots import *

####### system setup ####### ####### ####### ####### ####### ####### #######
# sys_name = 'VanderPol'
# sys_name = 'Pendubot'
sys_name = 'PendulumTrig'
# sys_name = 'PendulumRecast'
# sys_name = 'VirtualDubins'
# sys_name = 'VirtualDubins3d'

deg_ftrs = 2
deg_u = 1
rm_one = True
system = get_system(sys_name, deg_ftrs, deg_u, rm_one)

####### basis system analysis ####### ####### ####### ####### ####### #######
# stableSamples = system.sample_stable_inits(d=.2, num_grid=200)
# A, P, V = system.linearized_quadractic_V()

initial = system.random_sample(20)
# print(initial)
[final_states, final_Vs] = plot_traj(initial, system, int_horizon=500)
print([system.is_at_fixed_pt(i) for i in final_states])
####### ####### ####### ####### ####### ####### ####### ####### #######

####### model setup ####### ####### ####### ####### ####### ####### #######
# train_or_load = 'Train'
# train_or_load = 'Load'
# epochs = 50
# V, Vdot, system, model, P, u_weights = get_V(system, train_or_load,
#                                              epochs=epochs, verbose=True,
# validation_split=0, shuffle=True)

# plot3d(V, system, r_max=[np.pi,10], in_xo=True)
# plot3d(Vdot, system, r_max=[1.5*np.pi, 6])
# plot_traj(initial, system, int_horizon=10, V=V)
####### ####### ####### ####### ####### ####### ####### ####### #######


####### Verification ####### ####### ####### ####### ####### ####### #######

# init_root_threads = 25
# verify_via_bilinear(system)
# verify_via_equality(system, V)
# verify_via_variety(system, V, init_root_threads=init_root_threads)
####### ####### ####### ####### ####### ####### ####### ####### #######
