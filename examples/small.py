import numpy as np
from pdpf import PrimalDual

import matplotlib.pyplot as plt


c = np.array([150, 200, 300])
A = np.array([[3, 1, 2],
              [1, 3, 0],
              [0, 2, 4]])
b = np.array([60, 36, 48])


model = PrimalDual(c, A, b)
model.minimize(MEPS=1.0e-10, verbose=1)

print('---------- Results ----------')
print(model.res)

# optimal solution
print('The optimal value is {}'.format(model.res.x))

# dual solution
print('A dual solution is {}'.format(model.dual))

plt.figure(figsize=[6, 4])
plt.xlabel('iter')
plt.ylabel('objective value')
plt.plot(np.arange(1, model.res.nit+1), model.res.fun, marker='o')
plt.yscale('log')
plt.grid(True)
plt.rcParams["svg.fonttype"] = "none"
plt.savefig('small.svg')
