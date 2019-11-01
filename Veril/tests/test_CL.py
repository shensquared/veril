from Veril import ClosedControlledLoop
from Veril import Plants
import numpy as np

import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
from scipy.linalg import solve_lyapunov, solve_discrete_lyapunov
import pydrake
import numpy as np
from numpy.linalg import eig, inv
from pydrake.all import (MathematicalProgram, Polynomial,
                         Expression, SolutionResult,
                         Variables, Solve, Jacobian, Evaluate,
                         RealContinuousLyapunovEquation, Substitute,
                         MosekSolver)

options = {
    'plant_name': 'DoubleIntegrator',
    # 'plant_name': 'Pendulum',
    # 'plant_name': 'Satellite',
    'num_units': 4,
    'timesteps': 20,
    'num_samples': 100,
    'batch_size': 1,
    'epochs': 3,
    'dt': 1e-3,
    'obs_idx': None,
    'tag': 'fortestonly',
}

NN = ClosedControlledLoop.get_NNorCL(NNorCL='NN', **options)
CL = ClosedControlledLoop.get_NNorCL(**options)

test_time_steps = 1
test_num_samples = 1

plant = Plants.get(CL.plant_name, CL.dt, CL.obs_idx)
augedSys = ClosedControlledLoop.augmentedTanhPolySys(CL)
x = augedSys.sampleInitialStatesInclduingTanh(test_num_samples)
initx=x[:,0:2]
initc=x[:,2:6]
ext_in = np.zeros((test_num_samples, test_time_steps,1))


def test_call_CLsys(NN, CL, x):
    predicted = NN.predict([initx,initc,ext_in])
    print(predicted)
    rollout=ClosedControlledLoop.batchSim(CL, test_time_steps, init=
        [initx,initc],
       num_samples = test_num_samples)
    print(rollout)

test_call_CLsys(NN, CL, x)

s,f = augedSys.symbolicStatesAndDynamics()
env = dict(zip(s,x.T))
numerical_f = np.array([i.Evaluate(env) for i in f])
next_x = x+numerical_f*options['dt']
print(next_x)

