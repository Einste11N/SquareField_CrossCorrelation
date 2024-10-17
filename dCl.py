import torch as tc
tc.set_default_tensor_type(tc.DoubleTensor)

import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d
import camb
from copy import deepcopy


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
        
        # Adjust order for interpolations
        if len(z)<=5 : itp_order = 'linear'
        else: itp_order = 'cubic'

        # Interpolation functions of z, 
        # taking in list, numpy.ndarray or torch.tensorin returning to numpy.ndarray
        self.H_of_z = backgrounds.hubble_parameter                  # Hubble parameter, in unit 
        self.f_of_z = self.Growth_Rate_of_z(backgrounds, itp_order) # Logarithmic growth rate
        self.Xe = interp1d(z_array, Xe_of_z, kind = itp_order)      # Ionized franction Xe
        self.chi = interp1d(z_array, chi_of_z, kind = itp_order)    # Comoving distance chi

        # interpolation functions of k and z
        self.Pm = tc.tensor(Pm)                                     # Matter power spectrum
        
    def Growth_Rate_of_z(self, backgrounds, itp_order):
        '''
        Get the interpolation function for logarithmic growth rate f, 
        defined as f:=d(ln D)/d(ln a)
        '''
        # Since the growth rate almost does not vary with momentum scale, we fix kh=0.01 to get f
        f_of_z = backgrounds.get_redshift_evolution([0.01], self.z_array.tolist(), ['growth'])
        return interp1d(self.z_array, f_of_z.flatten(), kind = itp_order)

    def Pm_interpolation(self, kh_array, z_array):
        return interp2d_torch(self.kh_array, self.z_array, self.Pm, kh_array, z_array)


    def dCl(self, z, l, l1, l_min = 1, l_max = 1000, N_l = 1000, N_theta = 81):
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
        z = tc.tensor([z], dtype=tc.float64)
        l = tc.tensor([l], dtype=tc.float64)
        l1 = tc.tensor([l1], dtype=tc.float64)

        # Make the mesh grid for theta_1, |l_2|, and theta_2
        t1_list = tc.arange(N_theta, dtype=tc.float64) * 2 * tc.pi / N_theta
        t2_list = deepcopy(t1_list)
        l2_list = tc.linspace(l_min, l_max, N_l, dtype=tc.float64)
        t1, l2, t2 = tc.meshgrid(t1_list, t2_list, l2_list, indexing='ij')

        # Pre-define useful varibales and constants
        chi = self.chi(z)
        lsquare = l**2
        l1square = l1**2
        l2square = l2**2

        l_dot_l1 = Polar_dot(l, 0, l1, t1)
        l_dot_l2 = Polar_dot(l, 0, l2, t2)
        l1_dot_l2 = Polar_dot(l1, l2, t2, t2)

        l_m_l1_norm = tc.sqrt( lsquare + l1square - 2*l_dot_l1 )
        l_p_l2_norm = tc.sqrt( lsquare + l2square + 2*l_dot_l2 )
        l1_p_l2_norm = tc.sqrt( l1square + l2square + 2*l1_dot_l2 )
        l_m_l1_p_l2_norm = tc.sqrt( lsquare + l1square + l2square - 2*l_dot_l1 + 2*l_dot_l2 - 2*l1_dot_l2 )

        del(l_dot_l1)
        del(l_dot_l2)
        del(l1_dot_l2)

        theta_l_p_l2 = Evaluate_angle(2, l, 0, l2, t2)
        theta_l1_p_l2 = Evaluate_angle(2, l, 0, l2, t2)
        theta_l_m_l1_p_l2 = Evaluate_angle(3, l, 0, -l1, t1, l2, t2)


        Z_MEAN = 0.45 # mean redshift for HI observation
        FREQ_HI = 1420. # in unit MHz
        SIGMA_HI = 0.0115 * 1000. * (1. + Z_MEAN) / FREQ_HI
        SIGMA_KSZ = deepcopy(SIGMA_HI)

        ##################################################
        # Evaluate the integrand

        # Initialization
        dCl_tot = tc.empty_like(t2)

        # Contribution originate from each term in Wick Theorem
        # Term 5 
        dCl = - tc.cos(theta_l1_p_l2 - t2) # - (l_dot_l2 + l2square) / l_p_l2_norm / l2
        dCl *= self.Cross_Power(z, l1_p_l2_norm / chi, 'e', 'e')
        dCl *= self.Cross_Power(z, l_p_l2_norm / chi, 'v', 'HI')
        dCl *= self.Cross_Power(z, l2 / chi, 'v', 'HI')
        dCl_tot += dCl
        # Term 6 
        dCl = - tc.cos(theta_l1_p_l2 - t2) # - (l_dot_l2 + l2square) / l_p_l2_norm / l2
        dCl *= self.Cross_Power(z, l_m_l1_p_l2_norm / chi, 'e', 'e')
        dCl *= self.Cross_Power(z, l_p_l2_norm / chi, 'v', 'HI')
        dCl *= self.Cross_Power(z, l2 / chi, 'v', 'HI')
        dCl_tot += dCl
        # Term 8 
        dCl = tc.cos(theta_l_p_l2 - theta_l1_p_l2) # (l_dot_l1 + l_dot_l2 + l1_dot_l2 + l2square) / l_p_l2_norm / l1_p_l2_norm
        dCl *= self.Cross_Power(z, l_m_l1_p_l2_norm / chi, 'e', 'v')
        dCl *= self.Cross_Power(z, l_p_l2_norm / chi, 'e', 'HI')
        dCl *= self.Cross_Power(z, l2 / chi, 'v', 'HI')
        dCl_tot += dCl
        # Term 9
        dCl = tc.cos(theta_l_m_l1_p_l2 - t2) # (l_dot_l2 - l1_dot_l2 + l2square) / l_m_l1_p_l2_norm / l2
        dCl *= self.Cross_Power(z, l1_p_l2_norm / chi, 'e', 'v')
        dCl *= self.Cross_Power(z, l2 / chi, 'e', 'HI')
        dCl *= self.Cross_Power(z, l_p_l2_norm / chi, 'v', 'HI')
        dCl_tot += dCl
        # Term 10
        dCl = tc.cos(theta_l1_p_l2 - t2)# (l1_dot_l2 + l2square) / l1_p_l2_norm / l2
        dCl *= self.Cross_Power(z, l_p_l2_norm / chi, 'e', 'HI')
        dCl *= self.Cross_Power(z, l1_p_l2_norm / chi, 'e', 'v')
        dCl *= self.Cross_Power(z, l2 / chi, 'v', 'HI')
        dCl_tot += dCl
        # Term 11
        dCl *= - self.Cross_Power(z, l_p_l2_norm / chi, 'e', 'HI')
        dCl *= self.Cross_Power(z, l1_p_l2_norm / chi, 'v', 'v')
        dCl *= self.Cross_Power(z, l2 / chi, 'e', 'HI')
        dCl_tot += dCl
        # Term 13
        dCl = tc.cos(theta_l_m_l1_p_l2 - theta_l_p_l2) # (lsquare + l2square + 2*l_dot_l2 - l_dot_l1 - l1_dot_l2) / l_p_l2_norm / l_m_l1_p_l2_norm
        dCl *= self.Cross_Power(z, l2 / chi, 'e', 'HI')
        dCl *= self.Cross_Power(z, l_m_l1_p_l2_norm / chi, 'e', 'v')
        dCl *= self.Cross_Power(z, l1_p_l2_norm / chi, 'v', 'HI')
        dCl_tot += dCl
        # Term 14
        dCl *= - self.Cross_Power(z, l2 / chi, 'e', 'HI')
        dCl *= self.Cross_Power(z, l_m_l1_p_l2_norm / chi, 'v', 'v')
        dCl *= self.Cross_Power(z, l1_p_l2_norm / chi, 'e', 'HI')
        dCl_tot += dCl

        # The Beam Function
        dCl_tot *= self.Beam_kSZ(l_m_l1_norm, SIGMA_KSZ) * self.Beam_kSZ(l1, SIGMA_KSZ) * self.Beam_HI(l_p_l2_norm, SIGMA_HI) * self.Beam_HI(l2, SIGMA_HI)

        # The window functions
        dCl_tot *= l1 * l2 * self.F_kSZ(z)**2 * self.G_HI(z)**2 * self.dchi_by_dz(z)

        dCl_res = tc.sum(dCl_tot) * t1_list[1] * (l_max - l_min) / (N_l - 1)

        return dCl_res
    


    def dchi_by_dz(self, z):
        return sc.c / tc.tensor(self.H_of_z(z))

    def F_kSZ(self, z):
        return tc.tensor(self.Xe(z)) * (1+z)**2 / tc.tensor(self.chi(z))**2
    
    def G_HI(self, z):
        return 1 / (self.z_list[-1] - self.z_list[0]) / tc.tensor(self.chi(z))**2

    def Beam_kSZ(self, l, singma_kSZ):
        return tc.exp(-l**2 * singma_kSZ**2 / 2)
    
    def Beam_HI(self, l, singma_HI):
        return tc.exp(-l**2 * singma_HI**2 / 2)

    def Cross_Power(self, z, kh, b1, b2):
    
        if b1 not in ['e', 'v', 'HI'] or b2 not in ['e', 'v', 'HI']:
            print('b1 and b2 must be "e", "v" or "HI"')
            raise
        else:
            if b1 == 'e': B1 = self.bias_electron(kh, z)
            elif b1 == 'v': B1 = self.bias_velocity(kh, z)
            elif b1 == 'HI': B1 = self.bias_HI(kh, z)

            if b2 == 'e': B2 = self.bias_electron(kh, z)
            elif b2 == 'v': B2 = self.bias_velocity(kh, z)
            elif b2 == 'HI': B2 = self.bias_HI(kh, z)

        shape = kh.shape
        P =  B1 * B2 * (self.Pm_interpolation(kh.flatten(), z)).reshape(shape)
        return P
    
    def bias_electron(self, kh, z): # TO BE REVISED
        return kh/kh
    
    def bias_velocity(self, kh, z):
        b = 1/(1+z) * tc.tensor(self.H_of_z(z)) * tc.tensor(self.f_of_z(z)) / kh
        return b
    
    def bias_HI(self, kh, z): # TO BE REVISED
        return kh/kh
    



def Polar_dot(lx, thetax, ly, thetay):
    return lx * ly * np.cos(thetax - thetay)

def Evaluate_angle(N_vec, *vectors):
    
    if 2*N_vec != len(vectors):
        print('The input N_vec does not match the number of input vectors')
        raise
    else:
        l_x = 0.
        l_y = 0.
        for i in range(N_vec):
            l_x = l_x + vectors[2*i] * np.cos(vectors[2*i+1])
            l_y = l_y + vectors[2*i] * np.sin(vectors[2*i+1])
        
        return np.arctan2(l_y, l_x)  

def interp2d_torch(x, y, z, x_new, y_new, mode='bilinear'):
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
    # x = x.float()
    # y = y.float()
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

