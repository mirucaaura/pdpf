import numpy as np
from scipy.optimize import OptimizeResult


class PrimalDual:
    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
        self.res = None

    def make_Mq_from_cAb(self):
        m, k = self.A.shape
        m1 = np.hstack((np.zeros((m, m)), -self.A, self.b.reshape(m, -1)))
        m2 = np.hstack((self.A.T, np.zeros((k, k)), -self.c.reshape(k, -1)))
        m3 = np.append(np.append(-self.b, self.c), 0)
        M = np.vstack((m1, m2, m3))
        q = np.zeros(m + k + 1)
        return M, q


    def make_artProb_initialPoint(self):
        M, q = self.make_Mq_from_cAb()
        n, n = M.shape

        x0 = np.ones(n)
        mu0 = np.dot(q, x0) / (n + 1) + 1
        z0 = mu0 / x0
        r = z0 - np.dot(M, x0) - q
        qn1 = (n + 1) * mu0 

        MM = np.hstack((M, r.reshape((-1, 1))))
        MM = np.vstack((MM, np.append(-r, 0)))
        qq = np.append(q, qn1)
        xx0 = np.append(x0, 1)
        zz0 = np.append(z0, mu0)
        return MM, qq, xx0, zz0


    def binarysearch_theta(self, x, z, dx, dz, beta=0.5, precision=0.001):
        n = np.alen(x)

        th_low = 0.0
        th_high = 1.0
        if np.alen(-x[dx<0]/dx[dx<0]) > 0:
            th_high = min(th_high, np.min(-x[dx<0]/dx[dx<0]))
        if np.alen(-z[dz<0]/dz[dz<0]) > 0:
            th_high = min(th_high, np.min(-z[dz<0]/dz[dz<0]))

        x_low = x + th_low * dx
        z_low = z + th_low * dz
        x_high = x + th_high * dx
        z_high = z + th_high * dz
        mu_high = np.dot(x_high, z_high) / n
        if (beta * mu_high >= np.linalg.norm(x_high * z_high - mu_high * np.ones(n))):
            return th_high

        while th_high - th_low > precision:
            th_mid = (th_high + th_low) / 2
            x_mid = x + th_mid * dx
            z_mid = z + th_mid * dz
            mu_mid = np.dot(x_mid, z_mid) / n
            if (beta * mu_mid >= np.linalg.norm(x_mid * z_mid - mu_mid * np.ones(n))):
                th_low = th_mid
            else:
                th_high = th_mid

        return th_low


    def minimize(self, MEPS=1.0e-10, verbose=0):
        """
        Minimize a function using primal-dual path-following algorithm.

        Parameters
        ----------
        MEPS : float, optinal
            Stop when the gap between primal objective value and dual objective value
        verbose : int
            Integer value with logging level.

            0: no logging.

            1: display the value of theta and the value of objective function.
        """
        if verbose not in [0, 1]:
            raise ValueError("`verbose` must be in [0, 1, 2].")

        (M0, q0) = self.make_Mq_from_cAb()
        (M, q, x, z) = self.make_artProb_initialPoint()
        m, k = self.A.shape
        n, n = M.shape

        mu_log = []
        th_log = []

        count = 0
        mu = np.dot(z, x) / n
        while mu > MEPS:
            count += 1
            # predication steps
            delta = 0
            dx = np.linalg.solve(M + np.diag(z/x), delta* mu * (1/x) - z)
            dz = delta * mu * (1/x) - z - np.dot(np.diag(1/x), z * dx)
            th = self.binarysearch_theta(x, z, dx, dz)
            th_log.append(th)
            x = x + th * dx
            z = z + th * dz
            mu = np.dot(z, x) / n
            # correction steps
            delta = 1
            dx = np.linalg.solve(M + np.diag(z/x), delta * mu * (1/x) - z)
            dz = delta * mu * (1/x) - z - np.dot(np.diag(1/x), z * dx)
            x = x + dx
            z = z + dz
            mu = np.dot(z, x) / n
            mu_log.append(mu)

            if verbose >= 1:
                if count == 1:
                    print('{0:7s}{1:16s}{2:13s}'.format('Iter', ' theta', ' f(x)')
                            )
                print('{0:4d}\t{1:.4e}\t{2:.4e}'
                        .format(
                            count,
                            th,
                            mu))
    
        if x[n - 2] > MEPS:
            # print('Optimal solution:', x[m:m+k]/x[n-2], 'has found.')
            # print('Optimal value = ', np.dot(self.c, x[m:m+k]/x[n-2]))
            # print('Optimal solution (dual):', x[:m]/x[n-2], 'has found.')
            # print('Optimal value (dual) = ', np.dot(self.b, x[:m]/x[n-2]))
            
            self.res = OptimizeResult()
            self.res.x = x[m:m+k]/x[n-2]
            self.res.success = True
            self.res.message = 'Optimization terminated successfully.'
            self.res.status = 0
            self.res.fun = np.array(mu_log)
            self.res.nit = count
            self.res.dual = x[:m]/x[n-2]
        else:
            self.res = OptimizeResult()
            self.res.success = False
            self.message = 'NaN result encountered.'

        return np.array(th_log)
