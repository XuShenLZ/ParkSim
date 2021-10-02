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

        self.t = self.M.variable('t', mf.Domain.unbounded())
        self.w = self.M.variable('w', self.dim, mf.Domain.unbounded())

        # 2_norm(w) <= 1
        self.M.constraint(mf.Expr.vstack(1, self.w), mf.Domain.inQCone())

        for mu in mu_list:
            self.M.constraint( mf.Expr.sub(mf.Expr.dot( self.w, self.mu_e - mu), self.t), mf.Domain.greaterThan(0.0))
        
        self.M.objective(mf.ObjectiveSense.Maximize, self.t)

        self.M.solve()

        return self.t.level(), self.w.level()

class DecisionMaker(object):
    """
    Make a decision with the current weight vector
    """
    def __init__(self, beta=0):
        """
        instantiation
        """
        # Regularization coefficient
        self.beta = beta

    def solve(self, w: np.ndarray, phi: np.ndarray):
        """
        make decision
        """
        dim = phi.shape[1]

        self.M = mf.Model()
        self.p = self.M.variable('p', dim, mf.Domain.greaterThan(0.0))

        # Regularization
        self.r = self.M.variable('r', mf.Domain.greaterThan(0.0))
        self.M.constraint(mf.Var.vstack(self.r, self.p), mf.Domain.inQCone())

        # sum(p) = 1
        cons = self.M.constraint(mf.Expr.sum(self.p), mf.Domain.equalsTo(1.0))

        obj = mf.Expr.add(mf.Expr.dot(w, mf.Expr.mul(phi, self.p)), mf.Expr.mul(self.beta, self.r))

        self.M.objective(mf.ObjectiveSense.Minimize, obj)

        self.M.solve()

        return self.p.level(), self.p.dual(), cons.dual()
        