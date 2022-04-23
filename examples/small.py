import numpy as np
from pdpf import PrimalDual

c = np.array([150, 200, 300])
A = np.array([[3, 1, 2],
              [1, 3, 0],
              [0, 2, 4]])
b = np.array([60, 36, 48])


model = PrimalDual(c, A, b)
model.solve()

print(model.obj_log)
print(model.res)
