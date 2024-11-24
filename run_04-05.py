import torch as tc
tc.set_default_dtype(tc.float64)

import numpy as np
import dCl
from joblib import Parallel, delayed
import time

Nz = 10
zlist = tc.linspace(0.4, 0.5, 2)
l_list = tc.linspace(50, 200, 10)
pz_list = 10**tc.hstack([tc.linspace(-8,-4, 5)[:-1], tc.linspace(-4, -3, 3)[:-1], tc.linspace(-3, -1, 15)[:-1], tc.linspace(-1, 0, 5)])
lmax_list = tc.hstack([tc.linspace(800, 1000, 21)[:-1], 10**tc.linspace(3, 3.5, 5)])
l1_list = tc.hstack([ 10**tc.linspace(-4, 0, 5)[:-1], 10**tc.linspace(0, 1, 6)[:-1], tc.linspace(10, 300, 59), (10**tc.linspace(np.log10(300), np.log10(800), 13))[1:] ])

params = []
for zindex in range(Nz):
    for l in l_list:
        for pz_index in len(pz_list):
            for l1 in l1_list:
                params.append([zindex, l, pz_list[pz_index], lmax_list[pz_index], l1])

dCl_obj = dCl.Cl_kSZ2_HI2(zlist)

def compute_11_14(params):
    zindex, l, pz, lmax, l1 = params
    res5_beam_i = dCl_obj.dCl_lm_Term5(zindex, l, l1, pz, l_max=lmax)
    res6_beam_i =  dCl_obj.dCl_lp_Term6(zindex, l, l1, pz, l_max=lmax)
    return res5_beam_i, res6_beam_i


if __name__ == '__main__':
    
    # N_JOBS = 4
    # processed_list = Parallel(n_jobs=N_JOBS, prefer='threads')(delayed(compute_11_14)(i) for i in params)

    processed_list = []
    for i in params:
        res5, res6 = compute_11_14(i)
        print(i, '   ', end='\r')
        processed_list.append([res5, res6])

    res = np.array(processed_list).reshape([len(zlist), len(l_list), len(pz_list), len(l1_list), 2])
    np.save('res.npy', res)