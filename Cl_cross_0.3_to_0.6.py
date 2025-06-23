import torch as tc
tc.set_default_dtype(tc.float64)
tc.set_default_device('cuda')

import numpy as np
import dCl
from joblib import Parallel, delayed
import time

Nz = 4
zmin = 0.3
zmax = 0.6
zmin_text = '{:.2f}'.format(zmin)
zmax_text = '{:.2f}'.format(zmax)
zlist = tc.linspace(zmin, zmax, Nz)
dCl_obj = dCl.Cl_kSZ2_HI2(zlist)

print('redshift from ' + zmin_text + ' to ' + zmax_text)

l_list = tc.hstack([10**tc.linspace(2, np.log10(3000), 10), tc.linspace(4000., 8000., 5)])
t1_list = tc.linspace(0, tc.pi, 31)
l1_list = np.linspace(100, 10000, 45)


def compute(zindex, l, l1, t1):
    dC6, dC7_pre, dC8 = dCl_obj.dCl_Term_by_Term(zindex, l, l1, t1, Beam=False)
    dC7, approx7 = dCl_obj.Pvv_term7(zindex, l, l1, t1, Beam=False)
    return dC6, dC7, dC8, dC7_pre, approx7


N_JOBS = 4
do_parallel = True


zstart = 0
zend = len(zlist)

params = []
for l in l_list:
    for l1 in l1_list:
        params.append([l, l1])
length_p = len(params)
length_total = length_p * (zend - zstart)
length100 = length_total / 100
time0 = time.time()

for zindex in range(zstart, zend):
    res_zi = []
    for i, p in enumerate(params):
        l, l1 = p
        chi = dCl_obj.chi_of_z[zindex]

        if do_parallel:
            respack = Parallel(n_jobs=N_JOBS, prefer='threads')(
                delayed(compute)(zindex, l, l1, t1) 
                    for t1 in t1_list)
            res_zi.append(respack)

        else:
            for t1 in t1_list:
                dC6, dC7, dC8, dC7_pre, approx7 = compute(zindex, l, l1, t1)
                res_zi.append([dC6, dC7, dC8, dC7_pre, approx7])

        time_i = time.time() - time0
        total_index = (zindex - zstart) * length_p + i
        print(' ', total_index, ' of ', length_total, 
            ' percent: {:.3f}%'.format((total_index+1)/length100), 
            ', average velocity: {:.4f}% per min'.format((total_index+1)/length100 * 60. / time_i), 
            '    ', end='\r')
    
    print(' ')
    res_w = tc.tensor(res_zi).reshape([len(l_list), len(l1_list), len(t1_list), 5])
    print('saving data...')
    np.save('testdata/z_' + zmin_text + '_' + zmax_text + f'/dCl_cross_{zindex}.npy', res_w.to('cpu'))
