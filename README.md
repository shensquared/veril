# Veril

## Installation

1. Install [Mosek](https://www.mosek.com)

2. Install [Drake python binding](https://drake.mit.edu/python_bindings.html).

Note: due to the dependency on Mosek, Drake needs to be built from source. Also, be sure to set the DWITH_MOSEK flag on, e.g., `cmake -DWITH_GUROBI=ON -DWITH_MOSEK=ON ../drake`

3. `cd` to the veril root folder, and run
```python setup.py develop``` to install the veril package (and all other pip-installablle python dependecies).