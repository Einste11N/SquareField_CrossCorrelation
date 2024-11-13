import torch as tc
tc.set_default_dtype(tc.float64)

import numpy as np
import dCl
from joblib import Parallel, delayed
import time

Nz = 10
zlist = tc.linspace(0.4, 0.5, Nz)
l_list = tc.linspace(50, 200, 10)
pz_list = tc.hstack([tc.tensor([1e-8, 1e-6, 1e-4, 1e-3]), 10**tc.linspace(-2, -0.4, 9)])
l1_list = tc.hstack([ 10**tc.linspace(-4, 0, 5)[:-1], 10**tc.linspace(0, 1, 6)[:-1], tc.linspace(10, 300, 59), (10**tc.linspace(np.log10(300), np.log10(600), 10))[1:] ])

params = []
for zindex in range(Nz):
    for l in l_list:
        for pz in pz_list:
            for l1 in l1_list:
                params.append([zindex, l, pz, l1])

dCl_obj = dCl.Cl_kSZ2_HI2(zlist)

def compute_11_14(params):
    zindex, l, pz, l1 = params
    res11_nobeam_i, res11_beam_i = dCl_obj.dCl_lm_Term11(zindex, l, l1, pz, N_theta=210, dim=3, beam='both')
    res14_nobeam_i, res14_beam_i =  dCl_obj.dCl_lp_Term14(zindex, l, l1, pz, N_theta=210, dim=3, beam='both')
    return res11_beam_i, res14_beam_i, res11_nobeam_i, res14_nobeam_i


if __name__ == '__main__':
    
    N_JOBS = 8
    processed_list = Parallel(n_jobs=N_JOBS, prefer='threads')(delayed(compute_11_14)(i) for i in params)
    res = np.array(processed_list).reshape([len(zlist), len(l_list), len(pz_list), len(l1_list), 4])
    np.save('res.npy', res)