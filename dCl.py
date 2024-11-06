import torch as tc
tc.set_default_dtype(tc.float64)

import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d
import camb
from copy import deepcopy

import time


class Cl_kSZ2_HI2():

    def __init__(self, z_array, Tb = 1.8e-4, H0 = 67.75, ombh2 = 0.022):
        
        ##################################################s
        # Define the cosmological parameters
        params = camb.CAMBparams()
        params.set_cosmology(H0=H0, ombh2=ombh2)
        params.set_matter_power(redshifts = z_array, kmax=10, nonlinear=True)
        results = camb.get_results(params)
        backgrounds = camb.get_background(params)

        # Calculate the background evolution and results
        kh, z, Pm = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints = 500, var1='delta_tot', var2='delta_tot')
        Xe_of_z = np.array(backgrounds.get_background_redshift_evolution(z_array, ['x_e'], format='array')).flatten()
        chi_of_z = np.array(results.comoving_radial_distance(z_array))

        ##################################################
        # Store the variables that we are interested in

        # Constant scalars and arrays
        self.TCMB = params.TCMB     # CMB temperature 2.7K
        self.Tb = Tb                # HI brightness temperature, in unite mK
        self.kh_list = kh           # Total kh array that we are interested in
        self.kh_array = tc.tensor(kh)
        self.z_list = z             # Total redshift array that we are interested in
        self.z_array = tc.tensor(z)
        self.Pm = tc.tensor(Pm)     # Matter power spectrum
        self.Pm = tc.tensor(Pm)     # Matter power spectrum

        # Functions of redshift
        self.H_of_z = tc.tensor(backgrounds.hubble_parameter(z)) / sc.c     # Hubble parameter over c, in unit h/Mpc
        self.f_of_z = tc.tensor(                                            # Logarithmic growth rate
            backgrounds.get_redshift_evolution([0.01], z, ['growth']) ).flatten()
        self.Xe_of_z = tc.tensor(Xe_of_z)                                   # Ionized franction Xe
        self.chi_of_z = tc.tensor(chi_of_z)                                 # Comoving distance chi, in unit Mpc/h
        self.dchi_by_dz = 1. / self.H_of_z                                  # Comoving distance growth rate dchi/dz
        self.F_kSZ = self.Xe_of_z * (1+self.z_array)**2 / self.chi_of_z**2  # F_kSZ, propto visibility function of kSZ
        self.G_HI = 1 / (z[-1] - z[0]) / self.chi_of_z**2                   # G_HI, proptp window function of HI

        # Save the cosmological model, for checking the result
        self.results = results
        self.BGEvolution = backgrounds

        # Instruments' properties
        Z_MEAN = 0.45 # mean redshift for HI observation
        FREQ_HI = 1420. # in unit MHz
        self.SIGMA_HI = 0.0115 * 1000. * (1. + self.z_array) / FREQ_HI
        self.SIGMA_KSZ = deepcopy(self.SIGMA_HI)
        self.SIGMA_HI_MEAN = 0.0115 * 1000. * (1. + Z_MEAN) / FREQ_HI
        self.SIGMA_KSZ_MEAN = deepcopy(self.SIGMA_HI_MEAN)

        # Arrays used for matter power spectrum interpolation
        # adding infrared asymptotic behavior (P proportional to k)
        self.kh_array_itp, self.Pm_itp = self.Infrared_cutoff()

    def Infrared_cutoff(self, N_add = 5, cut_off = tc.tensor([1.e-8])):
        kh = self.kh_list
        z = self.z_list
        Pm = self.Pm

        k_array_extra = tc.linspace(0., kh[0], N_add)
        k_array_infrared = deepcopy(k_array_extra)
        k_array_infrared[0] = cut_off
        Pm_infared = k_array_infrared.repeat(len(z)).reshape([len(z), N_add]) * Pm[:, :1] / kh[0]

        k_tot = tc.hstack([k_array_extra, tc.tensor(kh[1:])])
        Pm_tot = tc.hstack([Pm_infared, Pm[:, 1:]])

        return k_tot, Pm_tot
        
    def Growth_Rate_of_z(self, backgrounds, itp_order):
        '''
        Get the interpolation function for logarithmic growth rate f, 
        defined as f:=d(ln D)/d(ln a)
        '''
        # Since the growth rate almost does not vary with momentum scale, we fix kh=0.01 to get f
        f_of_z = backgrounds.get_redshift_evolution([0.01], self.z_list, ['growth'])
        return interp1d(self.z_list, np.array(f_of_z).flatten(), kind = itp_order)
    
    def Power_matter_1d(self, kh, zindex):
        return torch_interp1d(self.kh_array_itp, (self.Pm_itp)[zindex], kh)

    def Beam_kSZ(self, l, zindex=0, use_mean = False):
        if use_mean:
            return tc.exp(-l**2 * self.SIGMA_KSZ_MEAN**2 / 2)
        else:
            return tc.exp(-l**2 * self.SIGMA_KSZ[zindex]**2 / 2)
    
    def Beam_HI(self, l, zindex=0, use_mean = False):
        if use_mean:
            return tc.exp(-l**2 * self.SIGMA_HI_MEAN**2 / 2)
        else:
            return tc.exp(-l**2 * self.SIGMA_HI[zindex]**2 / 2)

    def bias_electron(self, kh, zindex): # TO BE REVISED
        return kh/kh
        
    def bias_HI(self, kh, zindex): # TO BE REVISED
        return kh/kh
    
    def bias_velocity(self, kh, zindex, cut_off = tc.tensor([1.e-8])):
        z_dependence = 1/(1+self.z_array[zindex]) * self.H_of_z[zindex] * self.f_of_z[zindex]
        # cut off the divergence at infrared
        return tc.where(kh > cut_off, z_dependence / kh, z_dependence / cut_off)
       
    def integral_over_z(self, dCl_tot):
        # The window functions
        dCl_tot *= self.F_kSZ**2 * self.G_HI**2 * self.dchi_by_dz
        dCl_res = tc.trapz(dCl_tot, self.z_array, dim=-1)
        return dCl_res


    def dCl_l1(self, zi, l, l1, l_min = 1, l_max = 800, N_l = 1600, N_theta = 900, dim=3, theta = tc.pi / 2.):
        """Evaluare the integrand, dCl, as a function of z, l and l_1.

        Here we sum over theta_1, l_2, and theta_2. To get the final C_l result, one has to integrate dCl over chi and l_1, for a given l.

        Input
        -----
        `z` : float. 
            The redshift. 

        `l` : float. 
            The moment for C_l. Don't need to be an integer since we are in flat-sky approximation.

        `l1` : float.
            The norm of \\vec{l}_1.

        """
        ##################################################
        # Redefine the inputs as tc.tensors
        l = tc.tensor([l])
        l1 = tc.tensor([l1])

        # Make the mesh grid for theta_1, |l_2|, and theta_2
        t2_list = tc.arange(N_theta + 1) * 2 * tc.pi / N_theta
        l2_list = tc.hstack([tc.linspace(1e-4, l_min, 11)[:-1], tc.linspace(l_min, l_max, N_l)])

        if dim==3:
            t1_list = tc.linspace(0, tc.pi, N_theta//2 + 1)
            t1, l2, t2 = tc.meshgrid(t1_list, l2_list, t2_list, indexing='ij')
        elif dim==2:
            t1 = tc.tensor([theta])
            l2, t2 = tc.meshgrid(l2_list, t2_list, indexing='ij')
        else:
            print('dim can only be 3(default) or 2 by now')
            raise


        # Pre-define useful varibales and constants
        chi = self.chi_of_z[zi]
        lsquare = l**2
        l1square = l1**2
        l2square = l2**2

        l_dot_l1 = Polar_dot(l, 0., l1, t1)
        l_dot_l2 = Polar_dot(l, 0., l2, t2)
        l1_dot_l2 = Polar_dot(l1, t1, l2, t2)

        epsilon = tc.tensor([1e-8])
        k_l1_p_l2_norm = tc.sqrt( (l1square + l2square + 2*l1_dot_l2).abs() ) / chi + epsilon
        k_l_p_l2_norm = tc.sqrt( (lsquare + l2square + 2*l_dot_l2).abs() ) / chi + epsilon
        k_l_m_l1_p_l2_norm = tc.sqrt( (lsquare + l1square + l2square - 2*l_dot_l1 + 2*l_dot_l2 - 2*l1_dot_l2).abs() ) / chi + epsilon
        k_l2 = l2 / chi

        # Delete redundant variables to save memory
        del(l1_dot_l2)

        theta_l_p_l2 = Evaluate_angle(2, l, tc.tensor([0.]), l2, t2)
        theta_l1_p_l2 = Evaluate_angle(2, l, tc.tensor([0.]), l2, t2)
        theta_l_m_l1_p_l2 = Evaluate_angle(3, l, tc.tensor([0.]), -l1, t1, l2, t2)

        # Pre-calculate the matter power spectrum
        P_l1_p_l2_norm = self.Power_matter_1d(k_l1_p_l2_norm, zi)
        P_l_p_l2_norm = self.Power_matter_1d(k_l_p_l2_norm, zi)
        P_l2 = self.Power_matter_1d(k_l2, zi)
        P_l_m_l1_p_l2_norm = self.Power_matter_1d(k_l_m_l1_p_l2_norm, zi)
       

        ##################################################
        # Evaluate the integrand
        # Initialization
        dCl_tot = tc.zeros_like(t2)

        # Contribution originate from each term in Wick Theorem
        # Term 5 and Term 6
        dCl = - tc.cos(theta_l_p_l2 - t2)
        dCl *= P_l1_p_l2_norm * self.bias_electron(k_l1_p_l2_norm,zi)**2 + P_l_m_l1_p_l2_norm * self.bias_electron(k_l_m_l1_p_l2_norm,zi)**2
        dCl *= P_l_p_l2_norm        * self.bias_velocity(k_l_p_l2_norm,zi)      * self.bias_HI(k_l_p_l2_norm,zi)
        dCl *= P_l2                 * self.bias_velocity(k_l2,zi)               * self.bias_HI(k_l2,zi)
        dCl_tot += dCl
        # Term 8 and Term 10
        dCl = tc.cos(theta_l_p_l2 - theta_l1_p_l2) * P_l_m_l1_p_l2_norm \
                                    * self.bias_electron(k_l_m_l1_p_l2_norm,zi) * self.bias_velocity(k_l_m_l1_p_l2_norm,zi)
        dCl += tc.cos(theta_l1_p_l2 - t2) * P_l1_p_l2_norm \
                                    * self.bias_electron(k_l1_p_l2_norm,zi)     * self.bias_velocity(k_l1_p_l2_norm,zi)
        dCl *= P_l_p_l2_norm        * self.bias_electron(k_l_p_l2_norm,zi)      * self.bias_HI(k_l_p_l2_norm,zi)
        dCl *= P_l2                 * self.bias_velocity(k_l2,zi)               * self.bias_HI(k_l2,zi)
        dCl_tot += dCl
        # Term 9 and Term 13
        dCl = tc.cos(theta_l_m_l1_p_l2 - t2) * P_l1_p_l2_norm \
                                    * self.bias_electron(k_l1_p_l2_norm,zi)     * self.bias_velocity(k_l1_p_l2_norm,zi)
        dCl += tc.cos(theta_l_m_l1_p_l2 - theta_l_p_l2) * P_l_m_l1_p_l2_norm \
                                    * self.bias_electron(k_l_m_l1_p_l2_norm,zi) * self.bias_velocity(k_l_m_l1_p_l2_norm,zi)
        dCl *= P_l_p_l2_norm        * self.bias_velocity(k_l_p_l2_norm,zi)      * self.bias_HI(k_l_p_l2_norm,zi)
        dCl *= P_l2                 * self.bias_electron(k_l2,zi)               * self.bias_HI(k_l2,zi)
        dCl_tot += dCl 

        # Delete redundant variables to save memory
        del(P_l1_p_l2_norm, P_l_p_l2_norm, P_l2, P_l_m_l1_p_l2_norm, theta_l_p_l2, theta_l1_p_l2, theta_l_m_l1_p_l2)

        # The beam functions and the metric determinant contribution
        l_m_l1_norm = tc.sqrt( (lsquare + l1square - 2*l_dot_l1).abs() )
        l_p_l2_norm = tc.sqrt( (lsquare + l2square + 2*l_dot_l2).abs() )
        dCl_tot *= self.Beam_kSZ(l_m_l1_norm,zi) * self.Beam_kSZ(l1,zi) * self.Beam_HI(l_p_l2_norm,zi) * self.Beam_HI(l2,zi) * l1 * l2

        if dim==3:
            dCl_res = tc.trapz(tc.trapz(tc.trapz(dCl_tot, t2_list, dim=-1), l2_list, dim=-1), t1_list, dim=-1)
        else:
            dCl_res = tc.trapz(tc.trapz(dCl_tot, t2_list, dim=-1), l2_list, dim=-1)

        return dCl_res
    
    def dCl_lm_Term11(self, zi, l, lm, l_min = 1, l_max = 800, N_l = 1600, N_theta = 240, dim=3, theta = tc.pi / 2.):
        '''
        Integrand for Term 11, with parameter redefine ``lp = (l2 + l1) / 2``, and ``lm = (l2 - l1) / 2``
        '''
        l = tc.tensor([l])
        lm = tc.tensor([lm])

        # Make the mesh grid for theta_1, |l_2|, and theta_2
        tp_list = tc.arange(N_theta + 1) * 2 * tc.pi / N_theta
        lp_list = tc.hstack([(10**tc.linspace(-4, np.log10(l_min), 31))[:-1], tc.linspace(l_min, l_max, N_l)])

        if dim==3:
            tm_list = tc.linspace(0, tc.pi, N_theta//2 + 1)
            tm, lp, tp = tc.meshgrid(tm_list, lp_list, tp_list, indexing='ij')
        elif dim==2:
            tm = tc.tensor([theta])
            lp, tp = tc.meshgrid(lp_list, tp_list, indexing='ij')
        else:
            print('dim can only be 3(default) or 2 by now')
            raise


        # Pre-define useful varibales and constants
        chi = self.chi_of_z[zi]
        lsquare = l**2
        lmsquare = lm**2
        lpsquare = lp**2

        l_dot_lm = Polar_dot(l, 0., lm, tm)
        l_dot_lp = Polar_dot(l, 0., lp, tp)
        lm_dot_lp = Polar_dot(lm, tm, lp, tp)

        l_m_lp_p_lm_norm = tc.sqrt( (lsquare + lmsquare + lpsquare + 2*l_dot_lm - 2*l_dot_lp - 2*lm_dot_lp).abs() )
        lp_m_lm_norm = tc.sqrt( (lmsquare + lpsquare - 2*lm_dot_lp).abs() )
        l_p_lm_p_lp_norm = tc.sqrt( (lsquare + lmsquare + lpsquare + 2*l_dot_lm + 2*l_dot_lp + 2*lm_dot_lp).abs() )
        lp_p_lm_norm = tc.sqrt( (lmsquare + lpsquare + 2*lm_dot_lp).abs() )
        # Delete redundant variables to save memory
        del(l_dot_lm, l_dot_lp, lm_dot_lp)

        # Pre-Evaluate the k modes
        epsilon = tc.tensor([1e-8])
        k_l_p_lm_p_lp_norm = l_p_lm_p_lp_norm / chi + epsilon
        k_lm_p_lp_norm = lp_p_lm_norm / chi + epsilon
        k_2lp = 2 * lp / chi

        # Pre-calculate the matter power spectrum
        P_l_p_lm_p_lp_norm = self.Power_matter_1d(k_l_p_lm_p_lp_norm, zi)
        P_lm_p_lp_norm = self.Power_matter_1d(k_lm_p_lp_norm, zi)
        P_2lp = self.Power_matter_1d(k_2lp, zi)

        # Term 11 contribution
        dCl = - P_l_p_lm_p_lp_norm * self.bias_electron(k_l_p_lm_p_lp_norm,zi) * self.bias_HI(k_l_p_lm_p_lp_norm,zi)
        dCl *= P_lm_p_lp_norm  * self.bias_electron(k_lm_p_lp_norm,zi) * self.bias_HI(k_lm_p_lp_norm,zi)
        dCl *= P_2lp * self.bias_velocity(k_2lp, zi)**2

        # Delete redundant variables to save memory
        del(k_l_p_lm_p_lp_norm, k_lm_p_lp_norm, k_2lp)

        # The beam functions and the metric determinant contribution
        dCl *= self.Beam_kSZ(l_m_lp_p_lm_norm) * self.Beam_kSZ(lp_m_lm_norm) * self.Beam_HI(l_p_lm_p_lp_norm) * self.Beam_HI(lp_p_lm_norm) * lm * lp * 4

        if dim==3:
            dCl_res = tc.trapz(tc.trapz(tc.trapz(dCl, tp_list, dim=-1), lp_list, dim=-1), tm_list, dim=-1)
        else:
            dCl_res = tc.trapz(tc.trapz(dCl, tp_list, dim=-1), lp_list, dim=-1)

        return dCl_res

    def dCl_lp_Term14(self, zi, l, lp, l_min = 1, l_max = 800, N_l = 1600, N_theta = 240, dim=3, theta = tc.pi / 2.):
        '''
        Integrand for Term 14, with parameter redefine ``lp = (l2 + l1) / 2``, and ``Lm = (l - l1 + l2) / 2 = l / 2 + lm``
        '''
        l = tc.tensor([l])
        lp = tc.tensor([lp])

        # Make the mesh grid for theta_1, |l_2|, and theta_2
        Tm_list = tc.arange(N_theta + 1) * 2 * tc.pi / N_theta
        Lm_list = tc.hstack([(10**tc.linspace(-4, np.log10(l_min), 31))[:-1], tc.linspace(l_min, l_max, N_l)])

        if dim==3:
            tp_list = tc.linspace(0, tc.pi, N_theta//2 + 4)
            tp, Lm, Tm = tc.meshgrid(tp_list, Lm_list, Tm_list, indexing='ij')
        elif dim==2:
            tp = tc.tensor([theta])
            Lm, Tm = tc.meshgrid(Lm_list, Tm_list, indexing='ij')
        else:
            print('dim can only be 3(default) or 2 by now')
            raise

        # Pre-define useful varibales and constants
        chi = self.chi_of_z[zi]
        lsquare = l**2
        lpsquare = lp**2
        Lmsquare = Lm**2

        l_dot_lp = Polar_dot(l, 0., lp, tp)
        l_dot_Lm = Polar_dot(l, 0., Lm, Tm)
        lp_dot_Lm = Polar_dot(lp, tp, Lm, Tm)

        l_m_lp_p_lm_norm =  tc.sqrt( (lsquare + Lmsquare/4 + lpsquare + l_dot_Lm - 2*l_dot_lp - lp_dot_Lm).abs() )
        lp_m_lm_norm =      tc.sqrt( (lsquare + Lmsquare/4 + lpsquare - l_dot_Lm + 2*l_dot_lp - lp_dot_Lm).abs() )
        l_p_lm_p_lp_norm =  tc.sqrt( (lsquare + Lmsquare/4 + lpsquare + l_dot_Lm + 2*l_dot_lp + lp_dot_Lm).abs() )
        lp_p_lm_norm =      tc.sqrt( (lsquare + Lmsquare/4 + lpsquare - l_dot_Lm - 2*l_dot_lp + lp_dot_Lm).abs() )
        # Delete redundant variables to save memory
        del(l_dot_lp, l_dot_Lm, lp_dot_Lm)

        # Pre-Evaluate the k modes
        epsilon = tc.tensor([1e-8])
        k_l_p_lm_p_lp_norm = l_p_lm_p_lp_norm / chi + epsilon
        k_lm_p_lp_norm = lp_p_lm_norm / chi + epsilon
        k_2Lm = 2 * Lm / chi

        # Pre-calculate the matter power spectrum
        P_l_p_lm_p_lp_norm = self.Power_matter_1d(k_l_p_lm_p_lp_norm, zi)
        P_lm_p_lp_norm = self.Power_matter_1d(k_lm_p_lp_norm, zi)
        P_2Lm = self.Power_matter_1d(k_2Lm, zi)

        # Term 11 contribution
        dCl = - P_l_p_lm_p_lp_norm * self.bias_electron(k_l_p_lm_p_lp_norm,zi) * self.bias_HI(k_l_p_lm_p_lp_norm,zi)
        dCl *= P_lm_p_lp_norm  * self.bias_electron(k_lm_p_lp_norm,zi) * self.bias_HI(k_lm_p_lp_norm,zi)
        dCl *= P_2Lm * self.bias_velocity(k_2Lm, zi)**2

        # Delete redundant variables to save memory
        del(k_l_p_lm_p_lp_norm, k_lm_p_lp_norm, k_2Lm)

        # The beam functions and the metric determinant contribution
        dCl *= self.Beam_kSZ(l_m_lp_p_lm_norm) * self.Beam_kSZ(lp_m_lm_norm) * self.Beam_HI(l_p_lm_p_lp_norm) * self.Beam_HI(lp_p_lm_norm) * Lm * lp * 4

        if dim==3:
            dCl_res = tc.trapz(tc.trapz(tc.trapz(dCl, Tm_list, dim=-1), Lm_list, dim=-1), tp_list, dim=-1)
        else:
            dCl_res = tc.trapz(tc.trapz(dCl, Tm_list, dim=-1), Lm_list, dim=-1)

        return dCl_res




def Polar_dot(lx, thetax, ly, thetay):
    return lx * ly * np.cos(thetax - thetay)

def Evaluate_angle(N_vec, *vectors):

    if 2*N_vec != len(vectors):
        print('The input N_vec does not match the number of input vectors')
        raise
    else:
        # We need to do some adjustment on vectors to match the broadcast rule
        # In order to keep vectors unchanged, make a copy of them for calculation
        vec = deepcopy(vectors)

        l_x = 0.
        l_y = 0.
        for i in range(N_vec):
            l_x = l_x + vec[2*i] * tc.cos(vec[2*i+1])
            l_y = l_y + vec[2*i] * tc.sin(vec[2*i+1])
        
        return tc.atan2(l_y, l_x)
    
def torch_interp1d(x, y, x_query):

    indices = tc.searchsorted(x, x_query) - 1
    indices = tc.clamp(indices, 0, len(x) - 2)

    x0, x1 = x[indices], x[indices + 1]
    y0, y1 = y[indices], y[indices + 1]

    slope = (y1 - y0) / (x1 - x0)
    y_query = y0 + slope * (x_query - x0)
    
    return y_query

def torch_interp2d(x, y, z, x_new, y_new, mode='bilinear'):
    '''
    Interpolates 2D data over a grid using PyTorch, mimicking `scipy.interpolate.interp2d`.
    
    Parameters:
        x (torch.Tensor): 1D tensor of x coordinates (size: N).
        y (torch.Tensor): 1D tensor of y coordinates (size: M).
        z (torch.Tensor): 2D tensor of shape (M, N) representing the grid values.
        x_new (torch.Tensor): 1D tensor of new x coordinates for interpolation (size: N').
        y_new (torch.Tensor): 1D tensor of new y coordinates for interpolation (size: M').
        mode (str): Interpolation mode ('bilinear', 'nearest'). Defaults to 'bilinear'.
        
    Returns:
        torch.Tensor: Interpolated values at new (x_new, y_new) grid points.
    '''
    
    # Ensure the input tensors are of the correct shape

    z = z.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions (1, 1, M, N)
    
    # Create the meshgrid for new points (x_new, y_new)
    x_new_grid, y_new_grid = tc.meshgrid(x_new, y_new, indexing='ij')
    
    # Normalize new grid coordinates to range [-1, 1] (for grid_sample)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_new_norm = 2 * (x_new_grid - x_min) / (x_max - x_min) - 1
    y_new_norm = 2 * (y_new_grid - y_min) / (y_max - y_min) - 1
    
    # Stack and reshape the new coordinates into (1, H', W', 2) for grid_sample
    grid = tc.stack((x_new_norm, y_new_norm), dim=-1).unsqueeze(0)
    
    # Perform the interpolation using grid_sample
    interpolated = tc.nn.functional.grid_sample(z, grid, mode=mode, align_corners=True)
    
    # Remove the batch and channel dimensions and return the result
    return interpolated.squeeze()
