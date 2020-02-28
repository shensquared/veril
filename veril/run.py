# import time, argparse
import numpy as np
from veril.systems import get_system
from veril.sample_lyap import get_V
from veril.symbolic_verifier import verify_via_equality, verify_via_bilinear, \
    levelset_sos, global_vdot
from veril.sample_variety import verify_via_variety
from veril.util.plots import *

####### system setup ####### ####### ####### ####### ####### ####### #######
sys_name = 'VanderPol'
# sys_name = 'Pendubot'
# sys_name = 'PendulumRecast'
# sys_name = 'VirtualDubins'
# sys_name = 'VirtualDubins3d'
# sys_name = 'DubinsRecast'

deg_ftrs = 3
deg_u = 1
rm_one = True
system = get_system(sys_name, deg_ftrs, deg_u, rm_one)

####### basis system analysis ####### ####### ####### ####### ####### #######
# stableSamples = system.sample_stable_inits(d=.2, num_grid=200)
# A, P, V = system.linearized_quadractic_V()

# initial = system.random_sample(20)
# print(initial)
# [final_states, final_Vs] = plot_traj(initial, system, int_horizon=30)
# print([system.is_at_fixed_pt(i) for i in final_states])
####### ####### ####### ####### ####### ####### ####### ####### #######

####### model setup ####### ####### ####### ####### ####### ####### #######
# train_or_load = 'Train'
train_or_load = 'Load'
system = get_system(sys_name, deg_ftrs, deg_u, rm_one)
epochs = 20
V, Vdot, system, model, P, u_weights = get_V(system, train_or_load, over_para=3,
                                             epochs=epochs, verbose=True,
                                             validation_split=0, shuffle=True)

# [plot3d(V, system, in_xo=False, slice_idx=i) for i in system.all_slices]
# [plot3d(Vdot, system, in_xo=False, slice_idx=i) for i in system.all_slices]

initial = system.random_sample(20)
# print(initial)
[final_states, final_Vs] = plot_traj(initial, system, int_horizon=30)
# print([system.is_at_fixed_pt(i) for i in final_states])
####### ####### ####### ####### ####### ####### ####### ####### #######


####### Verification ####### ####### ####### ####### ####### ####### #######
verify_via_bilinear(system)
verify_via_equality(system, V)
verify_via_variety(system, V, init_root_threads=20)
####### ####### ####### ####### ####### ####### ####### ####### #######
