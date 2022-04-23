import numpy as np
from pdpf import PrimalDual

import matplotlib.pyplot as plt

c = np.array([150, 200, 300])
A = np.array([[3, 1, 2],
              [1, 3, 0],
              [0, 2, 4]])
b = np.array([60, 36, 48])


model = PrimalDual(c, A, b)
model.minimize(MEPS=1.0e-10)

print(model.res)

plt.figure(figsize=[6, 4])
plt.xlabel('iter')
plt.ylabel('objective value')
plt.plot(model.res.fun, marker='o')
plt.grid(True)
plt.rcParams["svg.fonttype"] = "none"
plt.savefig('./figs/small.svg')
