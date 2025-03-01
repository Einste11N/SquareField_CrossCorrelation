import torch as tc
tc.set_default_dtype(tc.float64)

import numpy as np
import dCl
from joblib import Parallel, delayed
import time

Nz = 8
zmin = 0.3
zmax = 1.0
zmin_text = '{:.2f}'.format(zmin)
zmax_text = '{:.2f}'.format(zmax)
zlist = tc.linspace(zmin, zmax, Nz)
dCl_obj = dCl.Cl_kSZ2_HI2(zlist)

l_list = tc.hstack([tc.linspace(30, 300, 19), tc.linspace(330, 600, 10)])
pz_list = 10.**tc.linspace(-3,-1, 16)

params = []
for l in l_list:
    for pz in pz_list:
        params.append([l, pz])

print('redshift from ' + zmin_text + ' to ' + zmax_text)

def generate_l1_list():
    return tc.hstack([10.**tc.linspace(-4, 0, 5)[:-1], 
                        10.**tc.linspace(0, 1, 6)[:-1], 
                        tc.linspace(10, 300, 59)[:-1], 
                        10.**tc.linspace(np.log10(300), 3, 14)[:-1], 
                        10.**tc.linspace(3, 4, 10)])

def generate_lmax(l1, pz_chi):
    lmax = 2. * (l1 + pz_chi)
    lmax_cut = tc.ones_like(l1) * 150.
    return tc.max(lmax, lmax_cut)

def generate_Ntheta(l1, pz_chi):
    return 300 + 3 * tc.tensor(l1 /50 / pz_chi**(0.1), dtype=tc.int32)
    

def compute(zindex, l, pz, l1, lmax, Ntheta):
    res_beam, res_nobeam = dCl_obj.dCl_lm_Term5(zindex, l, l1, pz, l_max=lmax, N_theta=Ntheta, beam='both')
    return res_beam, res_nobeam

N_JOBS = 4
do_parallel = False

l1_list = generate_l1_list()

zstart = 1
zend = len(zlist)
length_p = len(params)
length_total = length_p * (zend - zstart)
length100 = length_total / 100
time0 = time.time()



for zindex in range(zstart, zend):
    res_beam = []
    res_nobeam = []
    for i, p in enumerate(params):
        l, pz = p
        chi = dCl_obj.chi_of_z[zindex]    
        lmax = generate_lmax(l1_list, pz*chi)
        Ntheta = generate_Ntheta(l1_list, pz*chi)

        
        if do_parallel:
            res_p = Parallel(n_jobs=N_JOBS, prefer='threads')(delayed(compute)(zindex, l, pz, l1_list[l1_index], lmax[l1_index], Ntheta[l1_index]) 
                                                        for l1_index in range(len(l1_list)))
            res_beam_p = [r[0] for r in res_p]
            res_nobeam_p = [r[1] for r in res_p]
            res_beam.append(res_beam_p)
            res_nobeam.append(res_nobeam_p)

        else:
            for l1_index in range(len(l1_list)):
                res1, res2 = compute(zindex, l, pz, l1_list[l1_index], lmax[l1_index], Ntheta[l1_index])
                res_beam.append(res1)
                res_nobeam.append(res2)

        time_i = time.time() - time0
        total_index = (zindex - zstart) * length_p + i
        print(' ', total_index, ' of ', length_total, ' percent: {:.3f}%'.format((total_index+1)/length100), 
            ', average velocity: {:.4f}% per min'.format((total_index+1)/length100 * 60. / time_i), 
            '    ', end='\r')
    
    print(' ')
    res_beam_w = tc.tensor(res_beam).reshape([len(l_list), len(pz_list), len(l1_list)])
    res_nobeam_w = tc.tensor(res_nobeam).reshape([len(l_list), len(pz_list), len(l1_list)])
    print('saving data...')
    np.save('Beam_data/z_' + zmin_text + '_' + zmax_text + f'/Cl_cross_{zindex}.npy', res_beam_w)
    np.save(f'NoBeam_data/z_' + zmin_text + '_' + zmax_text + f'/Cl_cross_{zindex}.npy', res_nobeam_w)
