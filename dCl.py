import torch as tc
tc.set_default_dtype(tc.float64)

import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d
import camb
from copy import deepcopy


# NE is given by $\sigma_T \bar{n}_{b,0}$ with
# ------------------------
# sigma_T = 6.652e-29 m^2
# n_b0    =  0.2515 / m^3  from CMB
#            0.2485 / m^3  from BBN
NE = 5.13e-7  # in unit 1/Mpc


class Cl_kSZ2_HI2():

    def __init__(self, z_array, Tb = 1.8e-1, OmHIh = 2.45e-4, H0 = 67.75, ombh2 = 0.022):
        
        ##################################################s
        # Define the cosmological parameters
        params = camb.CAMBparams()
        params.set_cosmology(H0=H0, ombh2=ombh2)
        params.set_matter_power(redshifts = z_array, kmax=10, nonlinear=True)
        results = camb.get_results(params)
        backgrounds = camb.get_background(params)

        # Calculate the background evolution and results
        kh, z, Pm = results.get_matter_power_spectrum(minkh=1e-4, maxkh=100, npoints = 350, var1='delta_tot', var2='delta_tot')
        Xe_of_z = np.array(backgrounds.get_background_redshift_evolution(z_array, ['x_e'], format='array')).flatten()
        f_of_z = np.array(backgrounds.get_redshift_evolution([0.01], z, ['growth'])).flatten()
        chi_of_z = np.array(results.comoving_radial_distance(z_array))

        ##################################################
        # Store the variables that we are interested in

        # Constant scalars and arrays
        self.TCMB = params.TCMB     # CMB temperature 2.7K
        self.kh_list = kh           # Total kh array that we are interested in
        self.kh_array = tc.tensor(kh)
        self.z_list = z             # Total redshift array that we are interested in
        self.z_array = tc.tensor(z)
        self.Pm = tc.tensor(Pm)     # Matter power spectrum

        # Functions of redshift
        self.H_of_z = tc.tensor(                                            # Hubble parameter over c, in unit Mpc
            backgrounds.hubble_parameter(z)) / (sc.c/1000.)                 
        self.f_of_z = tc.tensor(f_of_z)                                     # Logarithmic growth rate
        self.Xe_of_z = tc.tensor(Xe_of_z)                                   # Ionized franction Xe
        self.chi_of_z = tc.tensor(chi_of_z)                                 # Comoving distance chi, in unit Mpc
        self.dchi_by_dz = 1. / self.H_of_z                                  # Comoving distance growth rate dchi/dz
        self.bv_of_z = 1/(1+self.z_array) * self.H_of_z * self.f_of_z       # aHf, z-dependence part of velocity bias

        # Redshift dependence of kSZ
        self.dtau_dchi = NE * self.Xe_of_z * (1 + self.z_array)**2          # dtau / dchi
        self.tau_of_z = self.evaluate_tau(H0, ombh2)                        # tau, optical depth     
        self.visibility_of_z = self.dtau_dchi * tc.exp(-self.tau_of_z)      # Visibility function g(z) of kSZ effect, g = dtau/dchi * exp(-tau)
        self.F_kSZ = self.visibility_of_z / self.chi_of_z**2                # F_kSZ, propto visibility function of kSZ
        
        # Redshift dependence of HI
        self.Tb_of_z = Tb * OmHIh * (1+self.z_array)**2 * H0 / self.H_of_z  # HI brightness temperature, in unite K
        self.W_HI = self.Tb_of_z**2 / self.dchi_by_dz / (z[-1] - z[0])      # Window function of HI observation
        self.G_HI = self.W_HI / self.chi_of_z**2                            # G_HI, proptp window function of HI
        self.G_HI_auto = (self.W_HI / self.chi_of_z)**2                     # G_HI_auto for HI square field auto correlation

        # Save the cosmological model, for checking the result
        self.results = results
        self.BGEvolution = backgrounds

        # Instruments' properties
        Z_MEAN = 0.45 # mean redshift for HI observation
        FREQ_HI = 1420. # in unit MHz
        self.SIGMA_HI2 = (0.0115 * 1000. * (1. + self.z_array) / FREQ_HI)**2
        self.SIGMA_KSZ2 = (np.pi/180 / 60)**2 / 8 / np.log(2)
        self.SIGMA_HI_MEAN2 = (0.0115 * 1000. * (1. + Z_MEAN) / FREQ_HI)**2
        self.SIGMA_KSZ_MEAN2 = (np.pi/180 / 60)**2 / 8 / np.log(2)

        # Arrays used for matter power spectrum interpolation
        # adding infrared asymptotic behavior (P proportional to k)
        self.kh_array_itp, self.Pm_itp = self.cutoff()

    def evaluate_tau(self, H0, ombh2):
        obj = Cl_kSZ2(zmax=self.z_array[-1], Nz=40, H0=H0, ombh2=ombh2)
        tau = torch_interp1d(obj.z_array, obj.tau_of_z, self.z_array)
        return tau

    def cutoff(self, N_add = 25, ir_cut = 1.e-8, uv_cut = 1.e4):
        kh = self.kh_list
        z = self.z_list
        Pm = self.Pm

        k_ir = tc.tensor(np.geomspace(ir_cut, kh[0], N_add))
        Pm_ir = Pm[:, :1] * k_ir.repeat(len(z)).reshape([len(z), N_add]) / kh[0]

        k_uv = tc.tensor(np.geomspace(kh[-1], uv_cut, N_add))
        Pm_uv = Pm[:, -1:] * (k_uv.repeat(len(z)).reshape([len(z), N_add]) / kh[-1])**(-3)

        k_tot = tc.hstack([k_ir, tc.tensor(kh[1:-1]), k_uv])
        Pm_tot = tc.hstack([Pm_ir, Pm[:, 1:-1], Pm_uv])

        return k_tot, Pm_tot

    def Power_matter_1d(self, kh, zindex, ir_cut = 1.e-8, uv_cut = 1.e4):
        itp = 10.**torch_interp1d(tc.log10(self.kh_array_itp), tc.log10((self.Pm_itp)[zindex]), tc.log10(kh))
        power = tc.where(tc.logical_and(kh>ir_cut, kh<uv_cut), itp, 0.)
        return power

    def Beam_kSZ(self, l, zindex=0, use_mean = False):
        if use_mean:
            return tc.exp(-l**2 * self.SIGMA_KSZ_MEAN2 / 2)
        else:
            return tc.exp(-l**2 * self.SIGMA_KSZ2 / 2)
    
    def Beam_HI(self, l, zindex=0, use_mean = False):
        if use_mean:
            return tc.exp(-l**2 * self.SIGMA_HI_MEAN2 / 2)
        else:
            return tc.exp(-l**2 * self.SIGMA_HI2[zindex] / 2)

    def bias_electron(self, kh, zindex): # TO BE REVISED
        return kh/kh
        
    def bias_HI(self, kh, zindex): # TO BE REVISED
        return kh/kh
    
    def bias_velocity(self, kh, zindex, cut_off = tc.tensor([1.e-8])):
        z_dependence = 1/(1+self.z_array[zindex]) * self.H_of_z[zindex] * self.f_of_z[zindex]
        return z_dependence / kh
        # cut off the divergence at infrared
        # return tc.where(kh > cut_off, z_dependence / kh, z_dependence / cut_off)
        

    def dCl_lm_Term5(self, zi, l, lm, pz=1e-8, l_min = 1, l_max = 500, N_l = 2000, theta_min = 0., theta_max = 2*tc.pi, N_theta = 300, beam=True):
        '''
            Integrand for Term 5, 9, 10, 11, with parameter redefine ``lp = (l2 + l1) / 2``, and ``lm = (l2 - l1) / 2``, or inversely ``l2 = lp + lm``, and ``l1 = lp - lm``

            Assumming both bias for electron and for HI are equal to 1, $b_v(k)=aHf b_e / k$
        '''
        l = tc.tensor([l])
        lm = tc.tensor([lm])
        pz = tc.tensor([pz])

        # Make the mesh grid for theta_1, |l_2|, and theta_2
        tp_list = tc.linspace(theta_min, theta_max, N_theta + 1)
        lp_list = tc.hstack([(10**tc.arange(-4, np.log10(l_min), 0.1))[:-1], tc.linspace(l_min, l_max, N_l)])

        tm_list = tc.hstack([tc.linspace(0, 0.9 * tc.pi, 25)[:-1], tc.linspace(0.9 * tc.pi, tc.pi, 6)])
        tm, lp, tp = tc.meshgrid(tm_list, lp_list, tp_list, indexing='ij')

        # Pre-define useful varibales and constants
        chisquare = self.chi_of_z[zi]**2
        lsquare = l**2
        lmsquare = lm**2
        lpsquare = lp**2
        pzsquare = pz**2

        l_dot_lm = Polar_dot(l, 0., lm, tm)
        l_dot_lp = Polar_dot(l, 0., lp, tp)
        lm_dot_lp = Polar_dot(lm, tm, lp, tp)

        # Pre-Evaluate the k modes
        k_l_p_lm_p_lp_norm= tc.sqrt( (lsquare + lmsquare + lpsquare + 2*l_dot_lm + 2*l_dot_lp + 2*lm_dot_lp) / chisquare + pzsquare )
        k_lm_p_lp_norm =    tc.sqrt( (lmsquare + lpsquare + 2*lm_dot_lp) / chisquare + pzsquare )
        k_2lp = tc.sqrt( 4 * lpsquare / chisquare + pzsquare)
        # Delete redundant variables to save memory
        # del(l_dot_lm, l_dot_lp, lm_dot_lp)

        # Pre-calculate the matter power spectrum
        P_l_p_lm_p_lp_norm = self.Power_matter_1d(k_l_p_lm_p_lp_norm, zi)
        P_lm_p_lp_norm = self.Power_matter_1d(k_lm_p_lp_norm, zi)
        P_2lp = self.Power_matter_1d(k_2lp, zi)

        ##################################################
        # Evaluate the integrand
        # Power sepctrum contribution
        dCl = P_l_p_lm_p_lp_norm * P_lm_p_lp_norm  * P_2lp
        # Geometric factor
        Geo  = 1 / (k_l_p_lm_p_lp_norm * k_lm_p_lp_norm)**2 # Term 5
        Geo -= 1 / (k_l_p_lm_p_lp_norm * k_2lp)**2          # Term 9
        Geo -= 1 / (k_lm_p_lp_norm     * k_2lp)**2          # Term 10
        Geo += 1 / k_2lp**4                                 # Term 11
        Geo *= pzsquare * self.bv_of_z[zi]**2
        
        dCl = dCl * Geo

        # Delete redundant variables to save memory
        del(k_l_p_lm_p_lp_norm, k_lm_p_lp_norm, k_2lp)

        ##################################################
        # The beam functions and the metric determinant contribution
        
        dCl_nobeam = dCl * 4 * lm * lp
        
        if beam=='both':
            l_m_lp_p_lm_norm =  tc.sqrt( (lsquare + lmsquare + lpsquare + 2*l_dot_lm - 2*l_dot_lp - 2*lm_dot_lp).abs() )
            lp_m_lm_norm =      tc.sqrt( (lmsquare + lpsquare - 2*lm_dot_lp).abs() )
            dCl_beam =          dCl_nobeam * self.Beam_HI(l) * self.Beam_kSZ(l_m_lp_p_lm_norm) * self.Beam_kSZ(lp_m_lm_norm) 
            dCl_res_beam =      tc.trapz(tc.trapz(tc.trapz(dCl_beam, tp_list, dim=-1), lp_list, dim=-1), tm_list, dim=-1)
            dCl_res_nobeam =    tc.trapz(tc.trapz(tc.trapz(dCl_nobeam, tp_list, dim=-1), lp_list, dim=-1), tm_list, dim=-1)
            return dCl_res_beam, dCl_res_nobeam
        elif beam:
            l_m_lp_p_lm_norm =  tc.sqrt( (lsquare + lmsquare + lpsquare + 2*l_dot_lm - 2*l_dot_lp - 2*lm_dot_lp).abs() )
            lp_m_lm_norm =      tc.sqrt( (lmsquare + lpsquare - 2*lm_dot_lp).abs() )
            dCl_beam =          dCl_nobeam * self.Beam_HI(l) * self.Beam_kSZ(l_m_lp_p_lm_norm) * self.Beam_kSZ(lp_m_lm_norm) 
            dCl_res_beam =      tc.trapz(tc.trapz(tc.trapz(dCl_beam, tp_list, dim=-1), lp_list, dim=-1), tm_list, dim=-1)
            return dCl_res_beam
        else:
            dCl_res_nobeam =    tc.trapz(tc.trapz(tc.trapz(dCl_nobeam, tp_list, dim=-1), lp_list, dim=-1), tm_list, dim=-1)
            return dCl_res_nobeam

    def dCl_lp_Term6(self, zi, l, lp, pz=1e-8, l_min = 1, l_max = 500, N_l = 2000, theta_min = 0., theta_max = 2*tc.pi, N_theta = 300, beam=True):
        '''
            Integrand for Term 6, 8, 13, 14, with parameter redefine ``lp = (l2 + l1) / 2``, and ``Lm = (l - l1 + l2) / 2 = l/2 + lm``, or inversely ``l2 = lp + lm = lp + Lm - l/2``, and ``l1 = lp - lm = lp - Lm + l/2``

            Assumming both bias for electron and for HI are equal to 1, $b_v(k)=aHf b_e / k$
        '''
        l = tc.tensor([l])
        lp = tc.tensor([lp])
        pz = tc.tensor([pz])

        # Make the mesh grid for theta_1, |l_2|, and theta_2
        Tm_list = tc.linspace(theta_min, theta_max, N_theta + 1)
        Lm_list = tc.hstack([(10**tc.arange(-4, np.log10(l_min), 0.1))[:-1], tc.linspace(l_min, l_max, N_l)])

        tp_list = tc.hstack([tc.linspace(0, 0.9 * tc.pi, 25)[:-1], tc.linspace(0.9 * tc.pi, tc.pi, 6)])
        tp, Lm, Tm = tc.meshgrid(tp_list, Lm_list, Tm_list, indexing='ij')

        # Pre-define useful varibales and constants
        chisquare = self.chi_of_z[zi]**2
        lsquare = l**2
        lpsquare = lp**2
        Lmsquare = Lm**2
        pzsquare = pz**2

        l_dot_lp = Polar_dot(l, 0., lp, tp)
        l_dot_Lm = Polar_dot(l, 0., Lm, Tm)
        lp_dot_Lm = Polar_dot(lp, tp, Lm, Tm)

        # Pre-Evaluate the k modes
        k_l_p_lm_p_lp_norm = tc.sqrt( (lsquare/4 + Lmsquare + lpsquare + l_dot_Lm + l_dot_lp + 2*lp_dot_Lm) / chisquare + pzsquare )
        k_lm_p_lp_norm =     tc.sqrt( (lsquare/4 + Lmsquare + lpsquare - l_dot_Lm - l_dot_lp + 2*lp_dot_Lm) / chisquare + pzsquare )
        k_2Lm = tc.sqrt( 4 * Lmsquare / chisquare + pzsquare )
        # Delete redundant variables to save memory
        # del(l_dot_lp, l_dot_Lm, lp_dot_Lm)

        # Pre-calculate the matter power spectrum
        P_l_p_lm_p_lp_norm = self.Power_matter_1d(k_l_p_lm_p_lp_norm, zi)
        P_lm_p_lp_norm = self.Power_matter_1d(k_lm_p_lp_norm, zi)
        P_2Lm = self.Power_matter_1d(k_2Lm, zi)

        ##################################################
        # Evaluate the integrand
        # Power sepctrum contribution
        dCl = P_l_p_lm_p_lp_norm * P_lm_p_lp_norm  * P_2Lm
        # Geometric factor
        Geo  = 1 / (k_l_p_lm_p_lp_norm * k_lm_p_lp_norm)**2 # Term 6
        Geo -= 1 / (k_lm_p_lp_norm     * k_2Lm)**2          # Term 8
        Geo -= 1 / (k_l_p_lm_p_lp_norm * k_2Lm)**2          # Term 13
        Geo += 1 / k_2Lm**4                                 # Term 14
        Geo *= pzsquare * self.bv_of_z[zi]**2
        
        dCl = dCl * Geo

        # Delete redundant variables to save memory
        del(k_l_p_lm_p_lp_norm, k_lm_p_lp_norm, k_2Lm)

        ##################################################
        # The beam functions and the metric determinant contribution
        dCl_nobeam = dCl * 4 * Lm * lp
        

        dCl_res_nobeam = tc.trapz(tc.trapz(tc.trapz(dCl_nobeam, Tm_list, dim=-1), Lm_list, dim=-1), tp_list, dim=-1)
        
        if beam=='both':
            l_m_lp_p_lm_norm =  tc.sqrt( (lsquare/4 + Lmsquare + lpsquare + l_dot_Lm - l_dot_lp - 2*lp_dot_Lm).abs() )
            lp_m_lm_norm =      tc.sqrt( (lsquare/4 + Lmsquare + lpsquare - l_dot_Lm + l_dot_lp - 2*lp_dot_Lm).abs() )
            dCl_beam =          dCl_nobeam * self.Beam_HI(l) * self.Beam_kSZ(l_m_lp_p_lm_norm) * self.Beam_kSZ(lp_m_lm_norm) 
            dCl_res_beam =      tc.trapz(tc.trapz(tc.trapz(dCl_beam, Tm_list, dim=-1), Lm_list, dim=-1), tp_list, dim=-1)
            dCl_res_nobeam =    tc.trapz(tc.trapz(tc.trapz(dCl_nobeam, Tm_list, dim=-1), Lm_list, dim=-1), tp_list, dim=-1)
            return dCl_res_beam, dCl_res_nobeam
        elif beam:
            l_m_lp_p_lm_norm =  tc.sqrt( (lsquare/4 + Lmsquare + lpsquare + l_dot_Lm - l_dot_lp - 2*lp_dot_Lm).abs() )
            lp_m_lm_norm =      tc.sqrt( (lsquare/4 + Lmsquare + lpsquare - l_dot_Lm + l_dot_lp - 2*lp_dot_Lm).abs() )
            dCl_beam =          dCl_nobeam * self.Beam_HI(l) * self.Beam_kSZ(l_m_lp_p_lm_norm) * self.Beam_kSZ(lp_m_lm_norm) 
            dCl_res_beam =      tc.trapz(tc.trapz(tc.trapz(dCl_beam, Tm_list, dim=-1), Lm_list, dim=-1), tp_list, dim=-1)
            return dCl_res_beam
        else:
            dCl_res_nobeam =    tc.trapz(tc.trapz(tc.trapz(dCl_nobeam, Tm_list, dim=-1), Lm_list, dim=-1), tp_list, dim=-1)
            return dCl_res_nobeam


    def dCl_lm_Term5_2d(self, zi, l, pz, lm, tm, l_min = 1, l_max = 500, N_l = 2000, theta_min = 0., theta_max = 2*tc.pi, N_theta = 300):
        '''
            Integrand for Term 5, 9, 10, 11, with parameter redefine ``lp = (l2 + l1) / 2``, and ``lm = (l2 - l1) / 2``, or inversely ``l2 = lp + lm``, and ``l1 = lp - lm``

            Assumming both bias for electron and for HI are equal to 1, $b_v(k)=aHf b_e / k$
        '''
        l = tc.tensor([l])
        pz = tc.tensor([pz])
        lm = tc.tensor([lm])
        tm = tc.tensor([tm])

        # Make the mesh grid for theta_1, |l_2|, and theta_2
        tp_list = tc.linspace(theta_min, theta_max, N_theta + 1)
        lp_list = tc.hstack([(10**tc.arange(-4, np.log10(l_min), 0.1))[:-1], tc.linspace(l_min, l_max, N_l)])

        lp, tp = tc.meshgrid(lp_list, tp_list, indexing='ij')

        # Pre-define useful varibales and constants
        chisquare = self.chi_of_z[zi]**2
        lsquare = l**2
        lmsquare = lm**2
        lpsquare = lp**2
        pzsquare = pz**2

        l_dot_lm = Polar_dot(l, 0., lm, tm)
        l_dot_lp = Polar_dot(l, 0., lp, tp)
        lm_dot_lp = Polar_dot(lm, tm, lp, tp)

        # Pre-Evaluate the k modes
        k_l_p_lm_p_lp_norm= tc.sqrt( (lsquare + lmsquare + lpsquare + 2*l_dot_lm + 2*l_dot_lp + 2*lm_dot_lp) / chisquare + pzsquare )
        k_lm_p_lp_norm =    tc.sqrt( (lmsquare + lpsquare + 2*lm_dot_lp) / chisquare + pzsquare )
        k_2lp = tc.sqrt( 4 * lpsquare / chisquare + pzsquare)
        # Delete redundant variables to save memory
        # del(l_dot_lm, l_dot_lp, lm_dot_lp)

        # Pre-calculate the matter power spectrum
        P_l_p_lm_p_lp_norm = self.Power_matter_1d(k_l_p_lm_p_lp_norm, zi)
        P_lm_p_lp_norm = self.Power_matter_1d(k_lm_p_lp_norm, zi)
        P_2lp = self.Power_matter_1d(k_2lp, zi)

        ##################################################
        # Evaluate the integrand
        # Power sepctrum contribution
        dCl = P_l_p_lm_p_lp_norm * P_lm_p_lp_norm  * P_2lp
        # Geometric factor
        Geo  = 1 / (k_l_p_lm_p_lp_norm * k_lm_p_lp_norm)**2 # Term 5
        Geo -= 1 / (k_l_p_lm_p_lp_norm * k_2lp)**2          # Term 9
        Geo -= 1 / (k_lm_p_lp_norm     * k_2lp)**2          # Term 10
        Geo += 1 / k_2lp**4                                 # Term 11
        Geo *= pzsquare * self.bv_of_z[zi]**2
        
        dCl = dCl * Geo

        ##################################################
        # The beam functions and the metric determinant contribution
        l_m_lp_p_lm_norm =  tc.sqrt( (lsquare + lmsquare + lpsquare + 2*l_dot_lm - 2*l_dot_lp - 2*lm_dot_lp).abs() )
        lp_m_lm_norm =      tc.sqrt( (lmsquare + lpsquare - 2*lm_dot_lp).abs() )
        dCl_nobeam = dCl * 4 * lm * lp
        dCl_beam = dCl_nobeam * self.Beam_HI(l) * self.Beam_kSZ(l_m_lp_p_lm_norm) * self.Beam_kSZ(lp_m_lm_norm) 

        dCl_res_beam = tc.trapz(tc.trapz(dCl_beam, tp_list, dim=-1), lp_list, dim=-1)

        return dCl_res_beam

    def dCl_lp_Term6_2d(self, zi, l, pz, lp, tp, l_min = 1, l_max = 500, N_l = 2000, theta_min = 0., theta_max = 2*tc.pi, N_theta = 300):
        '''
            Integrand for Term 6, 8, 13, 14, with parameter redefine ``lp = (l2 + l1) / 2``, and ``Lm = (l - l1 + l2) / 2 = l/2 + lm``, or inversely ``l2 = lp + lm = lp + Lm - l/2``, and ``l1 = lp - lm = lp - Lm + l/2``

            Assumming both bias for electron and for HI are equal to 1, $b_v(k)=aHf b_e / k$
        '''
        l = tc.tensor([l])
        pz = tc.tensor([pz])
        lp = tc.tensor([lp])
        tp = tc.tensor([tp])

        # Make the mesh grid for theta_1, |l_2|, and theta_2
        Tm_list = tc.linspace(theta_min, theta_max, N_theta + 1)
        Lm_list = tc.hstack([(10**tc.arange(-4, np.log10(l_min), 0.1))[:-1], tc.linspace(l_min, l_max, N_l)])

        Lm, Tm = tc.meshgrid(Lm_list, Tm_list, indexing='ij')

        # Pre-define useful varibales and constants
        chisquare = self.chi_of_z[zi]**2
        lsquare = l**2
        lpsquare = lp**2
        Lmsquare = Lm**2
        pzsquare = pz**2

        l_dot_lp = Polar_dot(l, 0., lp, tp)
        l_dot_Lm = Polar_dot(l, 0., Lm, Tm)
        lp_dot_Lm = Polar_dot(lp, tp, Lm, Tm)

        # Pre-Evaluate the k modes
        k_l_p_lm_p_lp_norm= tc.sqrt( (lsquare/4 + Lmsquare + lpsquare + l_dot_Lm + l_dot_lp + 2*lp_dot_Lm) / chisquare + pzsquare )
        k_lm_p_lp_norm =    tc.sqrt( (lsquare/4 + Lmsquare + lpsquare - l_dot_Lm - l_dot_lp + 2*lp_dot_Lm) / chisquare + pzsquare )
        k_2Lm = tc.sqrt( 4 * Lmsquare / chisquare + pzsquare )
        # Delete redundant variables to save memory
        # del(l_dot_lp, l_dot_Lm, lp_dot_Lm)

        # Pre-calculate the matter power spectrum
        P_l_p_lm_p_lp_norm = self.Power_matter_1d(k_l_p_lm_p_lp_norm, zi)
        P_lm_p_lp_norm = self.Power_matter_1d(k_lm_p_lp_norm, zi)
        P_2Lm = self.Power_matter_1d(k_2Lm, zi)

        ##################################################
        # Evaluate the integrand
        # Power sepctrum contribution
        dCl = P_l_p_lm_p_lp_norm * P_lm_p_lp_norm  * P_2Lm
        # Geometric factor
        Geo  = 1 / (k_l_p_lm_p_lp_norm * k_lm_p_lp_norm)**2 # Term 6
        Geo -= 1 / (k_lm_p_lp_norm     * k_2Lm)**2          # Term 8
        Geo -= 1 / (k_l_p_lm_p_lp_norm * k_2Lm)**2          # Term 13
        Geo += 1 / k_2Lm**4                                 # Term 14
        Geo *= pzsquare * self.bv_of_z[zi]**2
        
        dCl = dCl * Geo

        ##################################################
        # The beam functions and the metric determinant contribution
        l_m_lp_p_lm_norm =  tc.sqrt( (lsquare/4 + Lmsquare + lpsquare + l_dot_Lm - l_dot_lp - 2*lp_dot_Lm).abs() )
        lp_m_lm_norm =      tc.sqrt( (lsquare/4 + Lmsquare + lpsquare - l_dot_Lm + l_dot_lp - 2*lp_dot_Lm).abs() )
        dCl_nobeam = dCl * 4 * Lm * lp
        dCl_beam = dCl_nobeam * self.Beam_HI(l) * self.Beam_kSZ(l_m_lp_p_lm_norm) * self.Beam_kSZ(lp_m_lm_norm) 

        dCl_res_beam = tc.trapz(tc.trapz(dCl_beam, Tm_list, dim=-1), Lm_list, dim=-1)

        return dCl_res_beam

    def dCl_HI2(self, zi, l, l_min = 190, l_mid = None, l_max = None, N_l = 500, N_mu = 120, beam=True, noise = 0.):
        '''
            We denote k' as kk, theta_kk as the angle
            k = l / chi, theta_k = 0
            vector p = k - kk
        '''
        ##################################################
        # Redefine the inputs as tc.tensors and make the meshgrid
        chi = self.chi_of_z[zi]
        k = tc.tensor([l]) / chi
        if l_mid is None : l_mid = 2 * np.max([300., l])
        if l_max is None : l_max = 100. * chi

        kk_list = tc.hstack([tc.linspace(1e-4, 1, 11)[:-1], 
                            tc.linspace(1, l_min, 191)[:-1],
                            tc.linspace(l_min, l_mid, N_l)[:-1],
                            10**tc.linspace(np.log10(l_mid), np.log10(l_max), 50)]) / chi
        mu_list = tc.linspace(-1, 1, N_mu)

        kk, mu = tc.meshgrid(kk_list, mu_list, indexing='ij')
        k_m_kk_norm = k**2 + kk**2 - 2*k*kk*mu

        ##################################################
        # Evaluation
        Pnoise = noise / (1 + self.z_list[zi])**2 / self.Tb_of_z[zi]**2 / self.Beam_HI(l,zi)**2 
        dCl = kk**2 * (self.Power_matter_1d(kk, zi) + Pnoise) * (self.Power_matter_1d(k_m_kk_norm, zi) + Pnoise) / (2*tc.pi)**2

        ##################################################
        # Integral
        if beam=='both':
            dCl_beam = dCl * self.Beam_HI(l,zi)**4

            dCl_res_nobeam = tc.trapz(tc.trapz(dCl, mu_list, dim=-1), kk_list, dim=-1)
            dCl_res_beam = tc.trapz(tc.trapz(dCl_beam, mu_list, dim=-1), kk_list, dim=-1)
            return dCl_res_nobeam, dCl_res_beam

        elif beam:
            dCl_beam = dCl * self.Beam_HI(l,zi)**4
            return tc.trapz(tc.trapz(dCl_beam, mu_list, dim=-1), kk_list, dim=-1)
            
        else:
            return tc.trapz(tc.trapz(dCl, mu_list, dim=-1), kk_list, dim=-1)


    def dCl_lm_Term5_test(self, zi, l, lm, pz=1e-8, l_min = 1, l_max = 500, N_l = 2000, theta_min = 0., theta_max = 2*tc.pi, N_theta = 240, dim=2, theta = tc.pi / 3., debug=True, beam=True, resprint=True):
        '''
            Integrand for Term 5, 9, 10, 11, with parameter redefine ``lp = (l2 + l1) / 2``, and ``lm = (l2 - l1) / 2``, or inversely ``l2 = lp + lm``, and ``l1 = lp - lm``

            Assumming both bias for electron and for HI are equal to 1, $b_v(k)=aHf b_e / k$
        '''
        l = tc.tensor([l])
        lm = tc.tensor([lm])
        pz = tc.tensor([pz])

        # Make the mesh grid for theta_1, |l_2|, and theta_2
        tp_list = tc.linspace(theta_min, theta_max, N_theta + 1)
        lp_list = tc.hstack([(10**tc.arange(-4, np.log10(l_min), 0.1))[:-1], tc.linspace(l_min, l_max, N_l)])

        if dim==3:
            tm_list = tc.hstack([tc.linspace(0, 0.9 * tc.pi, 25)[:-1], tc.linspace(0.9 * tc.pi, tc.pi, 6)])
            tm, lp, tp = tc.meshgrid(tm_list, lp_list, tp_list, indexing='ij')
        elif dim==2:
            tm = tc.tensor([theta])
            lp, tp = tc.meshgrid(lp_list, tp_list, indexing='ij')
        else:
            print('dim can only be 3(default) or 2 by now')
            raise

        # Pre-define useful varibales and constants
        chisquare = self.chi_of_z[zi]**2
        lsquare = l**2
        lmsquare = lm**2
        lpsquare = lp**2
        pzsquare = pz**2

        l_dot_lm = Polar_dot(l, 0., lm, tm)
        l_dot_lp = Polar_dot(l, 0., lp, tp)
        lm_dot_lp = Polar_dot(lm, tm, lp, tp)

        # Pre-Evaluate the k modes
        k_l_p_lm_p_lp_norm= tc.sqrt( (lsquare + lmsquare + lpsquare + 2*l_dot_lm + 2*l_dot_lp + 2*lm_dot_lp) / chisquare + pzsquare )
        k_lm_p_lp_norm =    tc.sqrt( (lmsquare + lpsquare + 2*lm_dot_lp) / chisquare + pzsquare )
        k_2lp = tc.sqrt( 4 * lpsquare / chisquare + pzsquare)
        # Delete redundant variables to save memory
        # del(l_dot_lm, l_dot_lp, lm_dot_lp)

        # Pre-calculate the matter power spectrum
        P_l_p_lm_p_lp_norm = self.Power_matter_1d(k_l_p_lm_p_lp_norm, zi)
        P_lm_p_lp_norm = self.Power_matter_1d(k_lm_p_lp_norm, zi)
        P_2lp = self.Power_matter_1d(k_2lp, zi)

        ##################################################
        # Evaluate the integrand
        # Power sepctrum contribution
        dCl = P_l_p_lm_p_lp_norm * P_lm_p_lp_norm  * P_2lp
        # Geometric factor
        Geo  = 1 / (k_l_p_lm_p_lp_norm * k_lm_p_lp_norm)**2 # Term 5
        Geo -= 1 / (k_l_p_lm_p_lp_norm * k_2lp)**2          # Term 9
        Geo -= 1 / (k_lm_p_lp_norm     * k_2lp)**2          # Term 10
        Geo += 1 / k_2lp**4                                 # Term 11
        Geo *= pzsquare * self.bv_of_z[zi]**2
        
        dCl = dCl * Geo

        ##################################################
        # The beam functions and the metric determinant contribution
        l_m_lp_p_lm_norm =  tc.sqrt( (lsquare + lmsquare + lpsquare + 2*l_dot_lm - 2*l_dot_lp - 2*lm_dot_lp).abs() )
        lp_m_lm_norm =      tc.sqrt( (lmsquare + lpsquare - 2*lm_dot_lp).abs() )

        if beam=='both':
            dCl_nobeam = dCl * 4 * lm * lp
            dCl_beam = dCl_nobeam * self.Beam_kSZ(l_m_lp_p_lm_norm) * self.Beam_kSZ(lp_m_lm_norm) * self.Beam_HI(l)

            if dim==3:
                dCl_res_nobeam = tc.trapz(tc.trapz(tc.trapz(dCl_nobeam, tp_list, dim=-1), lp_list, dim=-1), tm_list, dim=-1)
                dCl_res_beam = tc.trapz(tc.trapz(tc.trapz(dCl_beam, tp_list, dim=-1), lp_list, dim=-1), tm_list, dim=-1)
            else:
                dCl_res_nobeam = tc.trapz(tc.trapz(dCl_nobeam, tp_list, dim=-1), lp_list, dim=-1)
                dCl_res_beam = tc.trapz(tc.trapz(dCl_beam, tp_list, dim=-1), lp_list, dim=-1)

            if resprint:
                print(dCl_res_nobeam, '    ', dCl_res_beam)
                
            if debug:
                return dCl_res_nobeam, dCl_res_beam, dCl_nobeam, dCl_beam, Geo, lp, tp
            else:
                return dCl_res_nobeam, dCl_res_beam

        elif beam:
            dCl *= 4 * lm * lp * self.Beam_kSZ(l_m_lp_p_lm_norm) * self.Beam_kSZ(lp_m_lm_norm) * self.Beam_HI(l)
        else:
            dCl *= 4 * lm * lp


        if dim==3:
            dCl_res = tc.trapz(tc.trapz(tc.trapz(dCl, tp_list, dim=-1), lp_list, dim=-1), tm_list, dim=-1)
        else:
            dCl_res = tc.trapz(tc.trapz(dCl, tp_list, dim=-1), lp_list, dim=-1)
        
        if resprint:
            print(dCl_res)
        
        if debug:
            return dCl_res, dCl, Geo, lp, tp
        else:
            return dCl_res

    def dCl_lp_Term6_test(self, zi, l, lp, pz=1e-8, l_min = 1, l_max = 500, N_l = 2000, theta_min = 0., theta_max = 2*tc.pi, N_theta = 240, dim=2, theta = tc.pi / 3., debug=True, beam=True, resprint=True):
        '''
            Integrand for Term 6, 8, 13, 14, with parameter redefine ``lp = (l2 + l1) / 2``, and ``Lm = (l - l1 + l2) / 2 = l/2 + lm``, or inversely ``l2 = lp + lm = lp + Lm - l/2``, and ``l1 = lp - lm = lp - Lm + l/2``

            Assumming both bias for electron and for HI are equal to 1, $b_v(k)=aHf b_e / k$
        '''
        l = tc.tensor([l])
        lp = tc.tensor([lp])
        pz = tc.tensor([pz])

        # Make the mesh grid for theta_1, |l_2|, and theta_2
        Tm_list = tc.linspace(theta_min, theta_max, N_theta + 1)
        Lm_list = tc.hstack([(10**tc.arange(-4, np.log10(l_min), 0.1))[:-1], tc.linspace(l_min, l_max, N_l)])

        if dim==3:
            tp_list = tc.hstack([tc.linspace(0, 0.9 * tc.pi, 25)[:-1], tc.linspace(0.9 * tc.pi, tc.pi, 6)])
            tp, Lm, Tm = tc.meshgrid(tp_list, Lm_list, Tm_list, indexing='ij')
        elif dim==2:
            tp = tc.tensor([theta])
            Lm, Tm = tc.meshgrid(Lm_list, Tm_list, indexing='ij')
        else:
            print('dim can only be 3(default) or 2 by now')
            raise

        # Pre-define useful varibales and constants
        chisquare = self.chi_of_z[zi]**2
        lsquare = l**2
        lpsquare = lp**2
        Lmsquare = Lm**2
        pzsquare = pz**2

        l_dot_lp = Polar_dot(l, 0., lp, tp)
        l_dot_Lm = Polar_dot(l, 0., Lm, Tm)
        lp_dot_Lm = Polar_dot(lp, tp, Lm, Tm)

        # Pre-Evaluate the k modes
        k_l_p_lm_p_lp_norm= tc.sqrt( (lsquare/4 + Lmsquare + lpsquare + l_dot_Lm + l_dot_lp + 2*lp_dot_Lm) / chisquare + pzsquare )
        k_lm_p_lp_norm =    tc.sqrt( (lsquare/4 + Lmsquare + lpsquare - l_dot_Lm - l_dot_lp + 2*lp_dot_Lm) / chisquare + pzsquare )
        k_2Lm = tc.sqrt( 4 * Lmsquare / chisquare + pzsquare )
        # Delete redundant variables to save memory
        # del(l_dot_lp, l_dot_Lm, lp_dot_Lm)

        # Pre-calculate the matter power spectrum
        P_l_p_lm_p_lp_norm = self.Power_matter_1d(k_l_p_lm_p_lp_norm, zi)
        P_lm_p_lp_norm = self.Power_matter_1d(k_lm_p_lp_norm, zi)
        P_2Lm = self.Power_matter_1d(k_2Lm, zi)

        ##################################################
        # Evaluate the integrand
        # Power sepctrum contribution
        dCl = P_l_p_lm_p_lp_norm * P_lm_p_lp_norm  * P_2Lm
        # Geometric factor
        Geo  = 1 / (k_l_p_lm_p_lp_norm * k_lm_p_lp_norm)**2 # Term 6
        Geo -= 1 / (k_lm_p_lp_norm     * k_2Lm)**2          # Term 8
        Geo -= 1 / (k_l_p_lm_p_lp_norm * k_2Lm)**2          # Term 13
        Geo += 1 / k_2Lm**4                                 # Term 14
        Geo *= pzsquare * self.bv_of_z[zi]**2
        
        dCl = dCl * Geo
        
        # Delete redundant variables to save memory
        del(k_l_p_lm_p_lp_norm, k_lm_p_lp_norm, k_2Lm)

        ##################################################
        # The beam functions and the metric determinant contribution
        l_m_lp_p_lm_norm =  tc.sqrt( (lsquare/4 + Lmsquare + lpsquare + l_dot_Lm - l_dot_lp - 2*lp_dot_Lm).abs() )
        lp_m_lm_norm =      tc.sqrt( (lsquare/4 + Lmsquare + lpsquare - l_dot_Lm + l_dot_lp - 2*lp_dot_Lm).abs() )

        if beam=='both':
            dCl_nobeam = dCl * 4 * Lm * lp
            dCl_beam = dCl_nobeam * self.Beam_kSZ(l_m_lp_p_lm_norm) * self.Beam_kSZ(lp_m_lm_norm) * self.Beam_HI(l)

            if dim==3:
                dCl_res_nobeam = tc.trapz(tc.trapz(tc.trapz(dCl_nobeam, Tm_list, dim=-1), Lm_list, dim=-1), tp_list, dim=-1)
                dCl_res_beam = tc.trapz(tc.trapz(tc.trapz(dCl_beam, Tm_list, dim=-1), Lm_list, dim=-1), tp_list, dim=-1)
            else:
                dCl_res_nobeam = tc.trapz(tc.trapz(dCl_nobeam, Tm_list, dim=-1), Lm_list, dim=-1)
                dCl_res_beam = tc.trapz(tc.trapz(dCl_beam, Tm_list, dim=-1), Lm_list, dim=-1)
            
            if resprint:
                print(dCl_res_nobeam, '    ', dCl_res_beam)

            if debug:
                return dCl_res_nobeam, dCl_res_beam, dCl_nobeam, dCl_beam, Geo, Lm, Tm
            else:
                return dCl_res_nobeam, dCl_res_beam

        elif beam:
            dCl *= 4 * Lm * lp * self.Beam_kSZ(l_m_lp_p_lm_norm) * self.Beam_kSZ(lp_m_lm_norm) * self.Beam_HI(l)
        else:
            dCl *= 4 * Lm * lp

        if dim==3:
            dCl_res = tc.trapz(tc.trapz(tc.trapz(dCl, Tm_list, dim=-1), Lm_list, dim=-1), tp_list, dim=-1)
        else:
            dCl_res = tc.trapz(tc.trapz(dCl, Tm_list, dim=-1), Lm_list, dim=-1)

        if resprint:
            print(dCl_res)
        if debug:
            return dCl_res, dCl, Geo, Lm, Tm
        else:
            return dCl_res


    def dCl_HI2_test(self, zi, l, l_min = 190, l_mid = None, l_max = None, N_l = 500, N_mu = 120, debug=True, beam=True, resprint=True):
        '''
            We denote k' as kk, theta_kk as the angle
            k = l / chi, theta_k = 0
            vector p = k - kk
        '''
        ##################################################
        # Redefine the inputs as tc.tensors and make the meshgrid
        chi = self.chi_of_z[zi]
        k = tc.tensor([l]) / chi
        if l_mid is None : l_mid = 2 * np.max([300., l])
        if l_max is None : l_max = 100. * chi

        kk_list = tc.hstack([tc.linspace(1e-4, 1, 11)[:-1], 
                            tc.linspace(1, l_min, 191)[:-1],
                            tc.linspace(l_min, l_mid, N_l)[:-1],
                            10**tc.linspace(np.log10(l_mid), np.log10(l_max), 50)]) / chi
        mu_list = tc.linspace(-1, 1, N_mu)

        kk, mu = tc.meshgrid(kk_list, mu_list, indexing='ij')
        k_m_kk_norm = k**2 + kk**2 - 2*k*kk*mu

        ##################################################
        # Evaluation
        dCl = kk**2 * self.Power_matter_1d(kk, zi) * self.Power_matter_1d(k_m_kk_norm, zi) / (2*tc.pi)**2

        ##################################################
        # Integral
        if beam=='both':
            dCl_beam = dCl * self.Beam_HI(l,zi)**2

            dCl_res_nobeam = tc.trapz(tc.trapz(dCl, mu_list, dim=-1), kk_list, dim=-1)
            dCl_res_beam = tc.trapz(tc.trapz(dCl_beam, mu_list, dim=-1), kk_list, dim=-1)

            if resprint: print(dCl_res_nobeam, dCl_res_beam)
            if debug: return dCl_res_nobeam, dCl_res_beam, dCl, dCl_beam, kk, mu
            else: return dCl_res_nobeam, dCl_res_beam

        elif beam:
            dCl = dCl * self.Beam_HI(l,zi)**2
        dCl_res = tc.trapz(tc.trapz(dCl, mu_list, dim=-1), kk_list, dim=-1)

        if resprint: print(dCl_res)
        if debug: return dCl_res, dCl, kk, mu
        else: return dCl_res



class Cl_kSZ2():

    def __init__(self, zmax = 10, zmin = 1e-3, Nz = 100, H0 = 67.75, ombh2 = 0.022, nonlinear = True):
        ##################################################s
        # Define the cosmological parameters

        z_array = tc.tensor(np.hstack([np.geomspace(zmin, 0.1, 10)[:-1], np.linspace(0.1, zmax, Nz, endpoint=True)]))
        params = camb.CAMBparams()
        params.set_cosmology(H0=H0, ombh2=ombh2)
        params.set_matter_power(redshifts = z_array, kmax=10, nonlinear=nonlinear)
        results = camb.get_results(params)
        backgrounds = camb.get_background(params)

        # Calculate the background evolution and results
        kh, z, Pm = results.get_matter_power_spectrum(minkh=1e-4, maxkh=100, npoints = 350, var1='delta_tot', var2='delta_tot')
        Xe_of_z = np.array(backgrounds.get_background_redshift_evolution(z_array, ['x_e'], format='array')).flatten()
        f_of_z = np.array(backgrounds.get_redshift_evolution([0.01], z, ['growth'])).flatten()
        chi_of_z = np.array(results.comoving_radial_distance(z_array))

        ##################################################
        # Store the variables that we are interested in

        # Constant scalars and arrays
        self.TCMB = params.TCMB     # CMB temperature 2.7K
        self.kh_list = kh           # Total kh array that we are interested in
        self.kh_array = tc.tensor(kh)
        self.z_list = z             # Total redshift array that we are interested in
        self.z_array = tc.tensor(z)
        self.Pm = tc.tensor(Pm)     # Matter power spectrum

        # Functions of redshift
        self.H_of_z = tc.tensor(
            backgrounds.hubble_parameter(z)) / (sc.c/1000.)             # Hubble parameter over c, in unit 1/Mpc
        self.f_of_z = tc.tensor(f_of_z)                                 # Logarithmic growth rate
        self.Xe_of_z = tc.tensor(Xe_of_z)                               # Ionized franction Xe
        self.chi_of_z = tc.tensor(chi_of_z)                             # Comoving distance chi, in unit Mpc
        self.dchi_by_dz = 1. / self.H_of_z                              # Comoving distance growth rate dchi/dz

        self.dtau_dchi = NE * self.Xe_of_z * (1 + self.z_array)**2      # dtau / dchi
        self.tau_of_z = tc.hstack([ tc.tensor([0.]),                    # tau, optical depth
            tc.cumulative_trapezoid(self.dtau_dchi * self.dchi_by_dz, self.z_array, dim=-1)])       
        self.g_of_kSZ = self.dtau_dchi * tc.exp(-self.tau_of_z)         # g_of_kSZ = dtau/dchi * exp(-tau)
        self.bv_of_z = 1/(1+self.z_array) * self.H_of_z * self.f_of_z   # aHf, z-dependence part of velocity bias, in unit 1/Mpc

        # Save the cosmological model, for checking the result
        self.results = results
        self.BGEvolution = backgrounds

        # Instruments' properties
        theta_FWHM = tc.tensor([1./60.]) # in unit deg
        self.SIGMA_KSZ2 = (np.pi/180 * theta_FWHM)**2 / 8 / np.log(2)
        self.SIGMA_KSZ_MEAN2 = (np.pi/180 * theta_FWHM)**2 / 8 / np.log(2)

        # Arrays used for matter power spectrum interpolation
        # adding infrared asymptotic behavior (P proportional to k)
        self.kh_array_itp, self.Pm_itp = self.cutoff()
        self.v_rms = self.compute_v_rms()

    def compute_v_rms(self):
        Pvv_k2 = self.bv_of_z[:,None]**2 * self.Pm_itp / (2*tc.pi**2)
        return tc.trapz(Pvv_k2, self.kh_array_itp, dim=-1)

    def cutoff(self, N_add = 25, ir_cut = 1.e-8, uv_cut = 1.e4):
        kh = self.kh_list
        z = self.z_list
        Pm = self.Pm

        k_ir = tc.tensor(np.geomspace(ir_cut, kh[0], N_add))
        Pm_ir = Pm[:, :1] * k_ir.repeat(len(z)).reshape([len(z), N_add]) / kh[0]

        k_uv = tc.tensor(np.geomspace(kh[-1], uv_cut, N_add))
        Pm_uv = Pm[:, -1:] * (k_uv.repeat(len(z)).reshape([len(z), N_add]) / kh[-1])**(-3)

        k_tot = tc.hstack([k_ir, tc.tensor(kh[1:-1]), k_uv])
        Pm_tot = tc.hstack([Pm_ir, Pm[:, 1:-1], Pm_uv])

        return k_tot, Pm_tot

    def Power_matter_1d(self, kh, zindex, ir_cut = 1.e-8, uv_cut = 1.e4):
        itp = 10.**torch_interp1d(tc.log10(self.kh_array_itp), tc.log10((self.Pm_itp)[zindex]), tc.log10(kh))
        power = tc.where(tc.logical_and(kh>ir_cut, kh<uv_cut), itp, 0.)
        return power

    def Beam_kSZ(self, l, zindex=0, use_mean = False):
        if use_mean:
            return tc.exp(-l**2 * self.SIGMA_KSZ_MEAN2 / 2)
        else:
            return tc.exp(-l**2 * self.SIGMA_KSZ2 / 2)

    def dCl_kSZ(self, zi, l, l_min = 190, l_max = None, N_l = 1100, N_mu = 200, beam=True):
        '''
            We denote k' as kk, theta_kk as the angle
            k = l / chi, theta_k = 0
            vector p = k - kk
        '''
        ##################################################
        # Redefine the inputs as tc.tensors and make the meshgrid
        chi = self.chi_of_z[zi]
        k = tc.tensor([l]) / chi

        if k > 100.:
            dCl = self.Power_matter_1d(k, zi) * self.v_rms[zi] / 3.
            dCl_beam = dCl * self.Beam_kSZ(l,zi)
            if beam=='both':    return dCl, dCl_beam
            elif beam:          return dCl_beam                
            else:               return dCl
        
        else:
            if l_max == None: l_max = 2 * np.max([200., l])
            kk_list = tc.hstack([tc.linspace(1e-4, 1, 11)[:-1], 
                                tc.linspace(1, l_min, 391)[:-1],
                                tc.linspace(l_min, l_max, N_l)]) / chi
            mu_list = tc.linspace(-1, 1, N_mu)

            kk, mu = tc.meshgrid(kk_list, mu_list, indexing='ij')
            theta_kk = tc.arccos(mu)
            k_m_kk_norm_square = k**2 + kk**2 - 2*k*kk*mu

            ##################################################
            # Evaluation
            dCl  = self.Power_matter_1d(kk, zi) * self.Power_matter_1d(tc.sqrt(k_m_kk_norm_square), zi)
            theta_p = Evaluate_angle(2, k, tc.tensor([0.]), -kk, theta_kk)
            sin_alpha = tc.sin(theta_p - theta_kk)
            # dCl *= k * (k - 2*kk*mu) * (1 - mu**2) / k_m_kk_norm_square
            dCl *= (k - 2*kk*mu) * sin_alpha**2 / k 

            dCl *= self.bv_of_z[zi]**2 / 2 / (2*tc.pi)**2

            ##################################################
            # Integral
            if beam=='both':
                dCl_beam = dCl * self.Beam_kSZ(l,zi)

                dCl_res_nobeam = tc.trapz(tc.trapz(dCl, mu_list, dim=-1), kk_list, dim=-1)
                dCl_res_beam = tc.trapz(tc.trapz(dCl_beam, mu_list, dim=-1), kk_list, dim=-1)
                return dCl_res_nobeam, dCl_res_beam

            elif beam:
                dCl_beam = dCl * self.Beam_kSZ(l,zi)
                return tc.trapz(tc.trapz(dCl_beam, mu_list, dim=-1), kk_list, dim=-1)
                
            else:
                return tc.trapz(tc.trapz(dCl, mu_list, dim=-1), kk_list, dim=-1)

    def Cl_kSZ(self, l, dCl_data = None, beam = True):
        if dCl_data == None:
            dCl_data = tc.empty_like(self.z_array)
            for zi in range(len(self.z_array)):
                dCl_data[zi] = self.dCl_kSZ(zi, l, beam = beam)
        
        dCl_dchi = dCl_data * (self.g_of_kSZ / self.chi_of_z)**2
        Cl = tc.trapz(dCl_dchi, self.chi_of_z, dim=-1)
        return Cl

    def Cl_kSZ2(self, l_list = tc.tensor(np.geomspace(10, 1e5, 101)), Cl_kSZ = None, ll_min = 10., ll_max = 1.e5, N_ll = 501, beam=True):
        if Cl_kSZ is None:
            Cl_kSZ = tc.empty_like(l_list)
            for i in range(len(l_list)):
                Cl_kSZ[i] = self.Cl_kSZ_test(l_list[i], beam = beam)

        lg_l_list = tc.log10(l_list)
        # lg_Cl_kSZ = tc.log10(Cl_kSZ)

        ll_list = tc.tensor(np.geomspace(ll_min, ll_max, N_ll))
        theta_list = tc.linspace(0, tc.pi, 100)
        l, ll, theta = tc.meshgrid(l_list, ll_list, theta_list)

        l_m_ll_norm = tc.sqrt(l**2 + ll**2 - 2*l*ll*tc.cos(theta))

        C1 = tc.empty_like(l)
        C2 = tc.empty_like(l)


        C1 = tc.where(tc.logical_and(l>=tc.tensor(ll_min), l<=tc.tensor(ll_max)),
                      torch_interp1d(lg_l_list, Cl_kSZ, tc.log10(l)), 
                      tc.tensor(0.))
        C2 = tc.where(tc.logical_and(l_m_ll_norm>=tc.tensor(ll_min), l_m_ll_norm<=tc.tensor(ll_max)), 
                      torch_interp1d(lg_l_list, Cl_kSZ, tc.log10(l_m_ll_norm)), 
                      tc.tensor(0.))

        CL = 2 * tc.trapz(tc.trapz(C1*C2*ll, theta_list, dim=-1), ll_list, dim=-1)

        return CL
    

    def dCl_kSZ_test(self, zi, l, l_min = 190, l_max = None, N_l = 1100, N_mu = 200, beam=True, resprint=True):
        ##################################################
        # Redefine the inputs as tc.tensors and make the meshgrid
        chi = self.chi_of_z[zi]
        k = tc.tensor([l]) / chi

        if l_max == None: l_max = 2 * np.max([200., l])

        kk_list = tc.hstack([tc.linspace(1e-4, 1, 11)[:-1], 
                             tc.linspace(1, l_min, 391)[:-1],
                             tc.linspace(l_min, l_max, N_l)]) / chi
        mu_list = tc.linspace(-1, 1, N_mu)

        kk, mu = tc.meshgrid(kk_list, mu_list, indexing='ij')
        theta_kk = tc.arccos(mu)
        k_m_kk_norm_square = k**2 + kk**2 - 2*k*kk*mu

        ##################################################
        # Evaluation
        dCl  = self.Power_matter_1d(kk, zi) * self.Power_matter_1d(tc.sqrt(k_m_kk_norm_square), zi)
        theta_p = Evaluate_angle(2, k, tc.tensor([0.]), -kk, theta_kk)
        sin_alpha = tc.sin(theta_p - theta_kk)
        # dCl *= k * (k - 2*kk*mu) * (1 - mu**2) / k_m_kk_norm_square
        dCl *= (k - 2*kk*mu) * sin_alpha**2 / k

        dCl *= self.bv_of_z[zi]**2 / 2 / (2*tc.pi)**2

        ##################################################
        # Integral
        if beam=='both':
            dCl_beam = dCl * self.Beam_kSZ(l,zi)

            dCl_res_nobeam = tc.trapz(tc.trapz(dCl, mu_list, dim=-1), kk_list, dim=-1)
            dCl_res_beam = tc.trapz(tc.trapz(dCl_beam, mu_list, dim=-1), kk_list, dim=-1)
            if resprint: print(dCl_res_nobeam, dCl_res_beam)
            return dCl_res_nobeam, dCl_res_beam, dCl, dCl_beam, kk, mu

        elif beam:
            dCl = dCl * self.Beam_kSZ(l,zi)
        res = tc.trapz(tc.trapz(dCl, mu_list, dim=-1), kk_list, dim=-1)
        if resprint: print(res)
        return tc.trapz(tc.trapz(dCl, mu_list, dim=-1), kk_list, dim=-1), dCl, kk, mu

    def Cl_kSZ_test(self, l, dCl_data = None, beam = True):
        if beam=='both':
            if dCl_data == None:
                dCl_data = tc.empty([2, len(self.z_array)])
                for zi in range(len(self.z_array)):
                    dCl_data[0,zi], dCl_data[1,zi] = self.dCl_kSZ(zi, l, beam = beam)
            
            dCl_dchi = dCl_data * (self.g_of_kSZ / self.chi_of_z)**2 # * self.dchi_by_dz
            Cl = tc.trapz(dCl_dchi, self.chi_of_z, dim=-1)
            return Cl[0], Cl[1]
        else:
            return self.Cl_kSZ(l, dCl_data=dCl_data, beam=beam)



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
    
def torch_interp1d(x, y, x_new):

    x_flat= x_new.flatten()
    xindices = tc.searchsorted(x, x_flat) - 1
    xindices = tc.clamp(xindices, 0, len(x) - 2)

    x0, x1 = x[xindices], x[xindices + 1]
    y0, y1 = y[xindices], y[xindices + 1]

    slope = (y1 - y0) / (x1 - x0)
    y_query = y0 + slope * (x_flat - x0)
    
    return y_query.reshape(x_new.shape)

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
