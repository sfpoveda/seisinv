from scipy import signal
import numpy as np
from scipy import stats

class MathOps:

    def __init__(self, vp, vs, rho):
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.alpha_gardner = 310
        self.beta_gardner = .25
    
    def calc_r(self, Z_data, mode='model'):
        if mode == 'model':
            r = []
            for i in range(1, len(Z_data)):
                num = Z_data[i] - Z_data[i-1]
                den = Z_data[i] + Z_data[i-1]
                r_temp = num/den
                r.append(r_temp)
            r = np.array(r)
            return r
        elif mode == 'well':
            temp = []
            for i in range(1, len(Z_data['Z'])):
                num = Z_data['Z'][i]-Z_data['Z'][i-1]
                den = Z_data['Z'][i]+Z_data['Z'][i-1]
                val = num/den
                temp.append(val)
                if i == len(Z_data['Z'])-1:
                    temp.append(temp[-1])
            Z_data['r'] = temp
            return Z_data

    def calc_G(self, m0, r0, r, step=.01, objective = 'Z'):
        G = np.zeros((len(m0), len(r0))) # 4x3 
        for j, row in enumerate(G): # for each row
            for i, col in enumerate(G[j]): # for each column
                if (i == j): # changes m_first
                    m_first = m0[j]
                    m_second = m0[j+1]
                    dZ = m_first*step
                    r_old = r0[i]
                    r_new = (m_second - (m_first+dZ))/(m_second + (m_first+dZ))
                    dr = r_old - r_new
                    cell_value = dr/dZ
                elif (i==(j-1)): # changes m_second
                    m_first = m0[j-1]
                    m_second = m0[j]
                    dZ = m_second*step
                    r_old = r0[i]
                    r_new = ((m_second+dZ) - m_first)/((m_second+dZ) + m_first)
                    dr = r_old - r_new
                    cell_value = dr/dZ
                else:
                    cell_value = 0
                G[j][i] = cell_value
        if objective == 'Z':
            G = G.T
        else:
            pass
        return G
    
    def calc_impedance(self):
        return self.rho*self.vp
    
    def calc_r_noisy(self, df, noise_weight):
        max_amp = abs(df['r']).max()*noise_weight
        noise = np.random.uniform(-max_amp,max_amp,len(df['r']))
        df['r_prime'] = df['r'] + noise
        return df
    
    def calc_rho(self, method='gardner', rho_b=None, Z=None, A=None, rho_m=None, rho_p=None, phi=None, Sp=None, rho_hc=None):
        # gardner density
        if method == 'gardner':
            rho = self.alpha_gardner*self.vp**self.beta_gardner
        # apparent density
        elif method == 'apparent':
            rho_e = rho_b*2*Z/A
            rho = 1.0704*rho_e - 0.1883
        # bulk density
        elif method == 'bulk':
            rho = rho_m * (1 - phi) + phi * rho_p
        # willie (bulk) density
        elif method == 'willie':
            rho = rho_m * (1 - phi) + phi * rho_p * Sp
        elif method == 'fluid':
            rho = Sp * rho_p + (1 - Sp) * rho_hc
        return rho
    
    def calc_vp(self):
        return (self.calc_impedance()/self.alpha_gardner)**(1/(1+self.beta_gardner))
    
    def wavelet_conv(self, df, r):
        # calculate convolved signal: wavelet + reflectivity profile
        points = 100
        a = 5
        vec2 = signal.ricker(points, a)
        df[f'waveform_{r}'] = np.convolve(df[r], vec2, 'same')
        return df
    
    def calc_sonic(self):
        return 1/self.vp
    
    def calc_dix(self, t, Vrms):
        V_int = []
        for i in range(1, len(Vrms)):
            v = np.sqrt((t[i]*Vrms[i]**2-t[i-1]*Vrms[i-1]**2)/(t[i]-t[i-1]))
            V_int.append(v)
        return V_int

    def calc_tx(self, t0, x, Vrms):
        return np.sqrt(t0**2 + x**2/Vrms**2)
    
    def calc_vs(self, mu, rho):
        return np.sqrt(mu/rho)
        
    def calc_mu(self, method='normal', mu_p=None, mu_m=None, phi=None): # Shear modulus
        if method == 'normal':
            mu = self.vs**2 * self.rho
        elif method == 'bulk': # solo si el fluido es viscoso (bulk)
            mu = (mu_p * mu_m) / (mu_m * phi + (1 - phi) * mu_p)
        elif method == 'non-viscous': # si el fluido no es viscoso (bulk)
            mu = mu_m / (1 - phi)
        return mu

    def calc_k(self, method='normal', k_p=None, k_m=None, phi=None): # bulk modulus
        if method == 'normal': 
            k = self.rho * (self.vp**2 - (4/3) * self.vs**2)
        elif method == 'bulk':
            k = (k_p * k_m) / (k_m * phi + (1 - phi) * k_p)
        return k
    
    def calc_lame_parameters(self):
        lambda_ = self.rho * (self.vp**2 - 2*self.vs**2)
        mu = self.rho * self.vs**2
        return lambda_, mu
    
    def calc_vp_vs_ratio(self):
        return self.vp/self.vs
    
    def calc_poisson(self):
        return (self.vp**2 - 2*self.vs**2)/(2 * (self.vp**2 - self.vs**2))
    
    def calc_coefficients_equiv_model(self):
        lambda_, mu = self.calc_lame_parameters()
        C = 1/(np.mean(1/(lambda_+2*mu)))
        term1 = 1/(np.mean(1/(lambda_ + 2*mu)))
        term2 = np.mean(lambda_ / (lambda_ + 2*mu))
        F = term1*term2
        L = 1/(np.mean(1/mu) )
        M = np.mean(mu)
        return C, F, L, M
    
    def calc_equivalent_vel(self):
        C, _, L, _ = self.calc_coefficients_equiv_model()
        rho0 = np.mean(self.rho)
        vp0 = np.sqrt(C/rho0)
        vs0 = np.sqrt(L/rho0)
        return vp0, vs0, rho0
    
    def calc_f(self, v, lambda_wave):
        f = v/lambda_wave
        return f
    
    def calc_porosity(self, method, t_log=None, t_m=None, rho_matrix=None, rho_rock=None, rho_pore=None, r=None):
        t_poro = t_log - t_m
        if method == 'sonic':
            phi = (t_log - t_m)/(t_poro - t_m)
        elif method == 'rho':
            phi = (rho_matrix - rho_rock)/(rho_matrix - rho_pore)
        elif method == 'theoric':
            phi = (4*r**2 - np.pi*r**2)/(4*r**2)
        return phi
    
    def calc_vol_shale(self, method, gamma_log=None, gamma_sand=None, gamma_log_shale=None, gamma_log_sand=None, SP_log=None, SP_sand=None, SP_shale=None):
        if method == 'gamma ray':
            v_shale = (gamma_log - gamma_sand)/(gamma_log_shale - gamma_log_sand)
        elif method == 'SP':
            v_shale = (SP_log - SP_sand)/(SP_shale - SP_sand)
        return v_shale
        
    def calc_fm_factor(self, method, porosity=None, R0=None, Rw=None):
        if method == 'porosity': 
            fm = 1/porosity**2
        elif method == 'resistivity':
            fm = R0/Rw
        return fm
    
    def calc_SP(self,R_fm, R_lodo, k):
        return -k*np.log(R_fm/R_lodo)
    
    def calc_saturation(self, method, Sw=None, R0 = None, Rt = None):
        if method == 'Sw':
            Shc = (1 - Sw)
        elif method == 'R':
            Sw = np.sqrt(R0/Rt)
            Shc = (1 - Sw)
        return Sw, Shc

class WaveAttenuation:

    def __init__(self, v, v0, f, f0):
        self.v = v
        self.v0 = v0
        self.f = f
        self.f0 = f0
        self.omega = 2*np.pi*f

    def calc_gamma(self):
        return np.log(self.v/self.v0)/(np.log(self.f/self.f0))
    
    def calc_Q(self):
        return 1/np.tan(self.calc_gamma()*np.pi)
    
    def calc_alpha(self):
        return self.omega/(2*self.calc_Q())

class Error:

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.len_y_true = len(y_true)
        self.len_y_pred = len(y_pred)
    
    def l1_error(self):
        # Calculate the absolute percentage error for each data point
        abs_percentage_error = np.abs((self.y_pred - self.y_true) / self.y_true)
        # Calculate the mean absolute percentage error as a percentage of n
        l1 = (np.sum(abs_percentage_error) / self.len_y_true) * 100
        return l1

    def l2_error(self):
        # Calculate the squared percentage error for each data point
        squared_percentage_error = ((self.y_true - self.y_pred) / self.y_true) ** 2
        # Calculate the mean squared percentage error and take the square root as a percentage of n
        l2 = (np.sqrt(np.sum(squared_percentage_error)) / self.len_y_true) * 100
        return l2
    
class Filtering:
    
    def remove_outliers(self, df, threshold=3):
        z_scores = stats.zscore(df)
        abs_z_scores = abs(z_scores)
        filtered_entries = (abs_z_scores < threshold).all(axis=1)  # Adjust the threshold as needed (e.g., 3 standard deviations)
        df_filtered = df[filtered_entries]
        df_filtered.reset_index(drop=True, inplace=True)
        return df_filtered

class ConversionTool(MathOps):
    
    def well_logs_2_seismic(self, lambda_wf_initial, lambda_wf_target):
        lambda_, mu = self.calc_lame_parameters()
        n_new_layers = int((len(self.rho)*lambda_wf_initial)/lambda_wf_target)
        ls_vp0 = []
        ls_vs0 = []
        ls_depth = []
        for i in range(n_new_layers):
            lambda_temp = lambda_[i * lambda_wf_target: (i + 1) * lambda_wf_target]
            mu_temp = mu[i * lambda_wf_target: (i + 1) * lambda_wf_target]
            rho_temp = self.rho[i * lambda_wf_target: (i + 1) * lambda_wf_target]
            # Calculate coefficients
            c = np.mean(1/(lambda_temp + 2*mu_temp))**(-1)
            f = (np.mean(1/(lambda_temp + 2*mu_temp))**(-1))*np.mean(lambda_temp / (lambda_temp + 2*mu_temp))
            l = np.mean(1/mu_temp)**(-1)
            m = np.mean(mu_temp)
            # Calculate average velocities
            vp0 = np.sqrt(c/np.mean(rho_temp))
            vs0 = np.sqrt(l/np.mean(rho_temp))
            ls_vp0.append(vp0)
            ls_vs0.append(vs0)
            print((i+1)*lambda_wf_target)
            ls_depth.append( (i+1)*lambda_wf_target )
        if (i == (n_new_layers-1)) and ( n_new_layers % (len(self.rho)/lambda_wf_target) != 0 ):
            lambda_temp = lambda_[(i + 1) * lambda_wf_target:]
            mu_temp = mu[(i + 1) * lambda_wf_target:]
            rho_temp = self.rho[(i + 1) * lambda_wf_target:]
            # Calculate coefficients
            c = np.mean(1/(lambda_temp + 2*mu_temp))**(-1)
            f = (np.mean(1/(lambda_temp + 2*mu_temp))**(-1))*np.mean(lambda_temp / (lambda_temp + 2*mu_temp))
            l = np.mean(1/mu_temp)**(-1)
            m = np.mean(mu_temp)
            # Calculate average velocities
            vp0 = np.sqrt(c/np.mean(rho_temp))
            vs0 = np.sqrt(l/np.mean(rho_temp))
            ls_vp0.append(vp0)
            ls_vs0.append(vs0)
            ls_depth.append(0)
        print('\n')
        print('Succesfully converted well log data resolution to seismic data resolution.')
        print(f'Data was downsampled from {lambda_wf_initial} to {lambda_wf_target}')
        return np.array(ls_vp0), np.array(ls_vs0), np.array(ls_depth)*lambda_wf_initial
    
    def SI_conversion(self, data, input_units):
        if input_units == 'km/s':
            data = data*1e+3
            print('Output units: m/s')
        elif input_units == 'ft':
            data = data*0.3048
            print('Output units: m')
        elif input_units == 'g/cm3':  
            data = data*1e+3
            print('Output units: kg/m3')
        elif input_units == 'micros/ft':
            data = data/((1e6)*(0.3048))
        return data