import time
import numpy as np
# import argparse
from veril.systems import get_system
from veril.sample_lyap import get_V

from veril.symbolic_verifier import verify_via_equality, verify_via_bilinear
from veril.sample_variety import verify_via_variety
from veril.util.plots import *

####### system setup ####### ####### ####### ####### ####### ####### #######
sys_name = 'VanderPol'
# sys_name = 'Pendubot'
# sys_name = 'PendulumTrig'
# sys_name = 'PendulumRecast'
degFeatures = 3
degU = 2
remove_one = True
system = get_system(sys_name, degFeatures, degU, remove_one=remove_one)
# stableSamples = system.sample_stable_inits(d=1, num_grid=10)

####### ####### ####### ####### ####### ####### ####### ####### #######


####### model setup ####### ####### ####### ####### ####### ####### #######
# train_or_load = 'Train'
train_or_load = 'Load'
epochs = 10
V, Vdot, system = get_V(system, train_or_load, epochs=epochs,
                        verbose=True, validation_split=0, shuffle=True)

# [plot3d(V, i, r_max = 1, level_sets=True) for i in system.all_slices]
# [plot3d(Vdot, i, level_sets=False, r_max=1.6)
#  for i in system.all_slices]

# initial = np.random.uniform(-np.pi, np.pi, (4, 2))
initial = system.random_sample(3)
plot_traj(initial, system, int_horizon=20, V=V)
####### ####### ####### ####### ####### ####### ####### ####### #######


####### Verification ####### ####### ####### ####### ####### ####### #######

init_root_threads = 25
# verify_via_bilinear(system)
# verify_via_equality(system, V)
verify_via_variety(system, V, init_root_threads=init_root_threads)
####### ####### ####### ####### ####### ####### ####### ####### #######
