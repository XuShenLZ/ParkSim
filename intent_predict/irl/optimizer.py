from typing import List
import numpy as np

import mosek.fusion as mf


class WeightOptimizer(object):
    """
    Optimize the weight vector
    """
    def __init__(self, mu_e: np.ndarray):
        """
        instantiation
        """
        self.mu_e = mu_e
        self.dim = self.mu_e.shape[0]

    def solve(self, mu_list: List[np.ndarray]):
        """
        solve the problem
        """
        self.M = mf.Model()
        # self.opti = ca.Opti()

        self.t = self.M.variable('t', mf.Domain.unbounded())
        self.w = self.M.variable('w', self.dim, mf.Domain.unbounded())

        # 2_norm(w) <= 1
        self.M.constraint(mf.Expr.vstack(1, self.w), mf.Domain.inQCone())

        for mu in mu_list:
            self.M.constraint( mf.Expr.sub(mf.Expr.dot( self.w, self.mu_e - mu), self.t), mf.Domain.greaterThan(0.0))
        
        self.M.objective(mf.ObjectiveSense.Maximize, self.t)

        self.M.solve()

        return self.t.level(), self.w.level()


        