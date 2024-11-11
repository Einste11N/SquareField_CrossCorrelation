import torch as tc
tc.set_default_dtype(tc.float64)

import numpy as np
import dCl
from joblib import Parallel, delayed
import time

Nz = 10
testclass = dCl.Cl_kSZ2_HI2(np.linspace(0.4, 0.5, Nz))

l_list = tc.linspace(50, 200, 10)
pz_list = tc.hstack([tc.tensor([1e-8, 1e-6, 1e-4, 1e-3]), 10**tc.linspace(-2, 0, 0.2)])
l1_list = tc.hstack([ 10**tc.linspace(-4, 0, 5)[:-1], 10**tc.linspace(0, 1, 6)[:-1], tc.linspace(10, 300, 59), (10**tc.linspace(np.log10(300), np.log10(600), 10))[1:] ])

def compute_11_14(params):
    l, l1, pz = params
    res11_nobeam_i, res11_beam_i = testclass.dCl_lm_Term11(0, l, l1, pz, N_theta=210, dim=3, beam='both')
    res14_nobeam_i, res14_beam_i =  testclass.dCl_lp_Term14(0, l, l1, pz, N_theta=210, dim=3, beam='both')
    return res11_nobeam_i, res11_beam_i, res14_nobeam_i, res14_beam_i


if __name__ == '__main__':
    N_JOBS = 4
    params = 0
    processed_list = Parallel(n_jobs=N_JOBS)(delayed(compute_11_14)(i) for i in params)
    print(0)