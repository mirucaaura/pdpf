# pdpf
python implementation of Primal-Dual Path-Following for Linear Programming

## Instrallation

`pdpf` is available on PyPI, and can be installed with

```shell
pip install git+https://github.com/mirucaaura/pdpf.git
```

`pdpf` has the following dependencies:

- Python >= 3.7
- NumPy >= 1.15
- SciPy >= 1.1.0

## Quickstart

We give an example on how to use this package.

First, import packages:

```python
import numpy as np
from pdpf import PrimalDual
import matplotlib.pyplot as plt
```

Define problem data:

```python
c = np.array([150, 200, 300])
A = np.array([[3, 1, 2],
              [1, 3, 0],
              [0, 2, 4]])
b = np.array([60, 36, 48])
```

Optimize:

```python
model = PrimalDual(c, A, b)
model.minimize()
```

Results can be shown by `model.res`:

```shell
>>> print(model.res)
     fun: array([4.33959524e-01, 1.92708144e-01, 7.28293862e-02, 2.23690150e-02,
       5.60288941e-03, 7.22804096e-04, 1.74412233e-05, 1.81881297e-08,
       1.77633299e-11])
 message: 'Optimization terminated successfully.'
     nit: 9
  status: 0
 success: True
       x: array([12.,  8.,  8.])
```

