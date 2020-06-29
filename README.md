# Veril

## Installation

1. Install [Mosek](https://www.mosek.com); version 9.1.9 is tested.

2. Install [Drake python binding](https://drake.mit.edu/python_bindings.html). Since Drake source SHA (45a43640636b9e3fd00f5f883c124d5b265df4a6) is tested, you may want to reset your HEAD to this commit before CMAKE.

```
git clone https://github.com/RobotLocomotion/drake.git
cd drake
git reset --hard 45a43640636b9e3fd00f5f883c124d5b265df4a6
cd ..
mkdir drake-build
cd drake-build
cmake ../drake
make -j
```
Note: due to the dependency on Mosek, Drake needs to be built from source. Also, be sure to set the DWITH_MOSEK flag on, e.g., `cmake -DWITH_GUROBI=ON -DWITH_MOSEK=ON ../drake`

3. `cd` to the veril root folder, and run
```python setup.py develop``` to install the veril package (and all other pip-installablle python dependecies).