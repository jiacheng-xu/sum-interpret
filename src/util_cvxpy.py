# CVXPY linear programming solver API
import cvxpy as cp
import numpy as np

def pnum(num):
    return "{:.2f}".format(num)

def compute_optimal_assignment(src_distributions, src_names, tgt_distb,tgt_name):
    src_distributions = [ x.numpy() for x in src_distributions]
    c = np.asarray(src_distributions)
    n = len(src_distributions)
    target = np.asarray(tgt_distb.numpy())

    x = cp.Variable((n))
    prob = cp.Problem(cp.Minimize( cp.sum(cp.abs(x@c - target))),
                 [cp.sum(x) == 1, x >= 0])
    prob.solve()
    logging.info(f"Optimal val: {pnum(prob.value)}")
    logging.info(f"Assignment: {[ src_names[idx] for idx in range(len(src_names))]}")
    logging.info(f"Assignment: {[ pnum(x.value[idx]) for idx in range(len(src_names))]}")