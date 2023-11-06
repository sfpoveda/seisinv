from scipy import signal
import numpy as np
from scipy import stats

class MathOps:

    def __init__(self, vp=None, vs=None, rho=None, depth=None, lambda_min_end=1, initial_dept=0):
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.depth = depth
        self.alpha_gardner = 310
        self.beta_gardner = .25
        self.lambda_min_end = lambda_min_end
        self.initial_dept = initial_dept
    
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
        intervals = []
        if self.lambda_min_end > 1: # if you want to transform layers to a specific wavelength (> 1 layer)
            n_equivalent_layers = int((len(self.vp) - 1) / self.lambda_min_end)
            for i in range(1, n_equivalent_layers):
                start = (i-1)*self.lambda_min_end
                end = i*self.lambda_min_end
                intervals.append([start, end])
        else: # if you want to obtain a single equivalente layer (1 layer)
            intervals.append([0, len(self.vp)-1])
        lambda_, mu = self.calc_lame_parameters()
        C_ls, F_ls, L_ls, M_ls, dept_ls, rho_ls = [], [], [], [], [], []
        for span in intervals:
            bottom, top = span[0], span[1] # start and end index
            if self.depth.iloc[bottom] not in dept_ls:
                dept_ls.append(self.depth.iloc[bottom])
            elif self.depth.iloc[top] not in dept_ls:
                dept_ls.append(self.depth.iloc[top])
            lambda_temp, mu_temp = lambda_[bottom:top], mu[bottom:top]
            rho = self.rho[bottom:top]
            C = 1/(np.mean(1/(lambda_temp+2*mu_temp)))
            term1 = 1/(np.mean(1/(lambda_temp + 2*mu_temp)))
            term2 = np.mean(lambda_temp / (lambda_temp + 2*mu_temp))
            F = term1*term2
            L = 1/(np.mean(1/mu_temp) )
            M = np.mean(mu_temp)
            C_ls.append(C)
            F_ls.append(F)
            L_ls.append(L)
            M_ls.append(M)
            rho_ls.append(np.mean(rho))
        # downsample the remaining last incomplete interval
        bottom, top = span[1], len(self.depth) - 1
        dept_ls.append(self.depth.iloc[top])
        lambda_temp, mu_temp = lambda_[bottom:top], mu[bottom:top]
        rho = self.rho[bottom:top]
        C = 1/(np.mean(1/(lambda_temp+2*mu_temp)))
        term1 = 1/(np.mean(1/(lambda_temp + 2*mu_temp)))
        term2 = np.mean(lambda_temp / (lambda_temp + 2*mu_temp))
        F = term1*term2
        L = 1/(np.mean(1/mu_temp) )
        M = np.mean(mu_temp)
        C_ls.append(C)
        F_ls.append(F)
        L_ls.append(L)
        M_ls.append(M)
        rho_ls.append(np.mean(rho))
        return np.array(C_ls), np.array(F_ls), np.array(L_ls), np.array(M_ls), np.array(dept_ls), np.array(rho_ls)
    
    def calc_equivalent_model(self):
        C_ls, F_ls, L_ls, M_ls, dept_ls, rho_ls = self.calc_coefficients_equiv_model()
        vp0 = np.sqrt(C_ls/rho_ls)
        vs0 = np.sqrt(L_ls/rho_ls)
        return vp0, vs0, rho_ls, dept_ls
    
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
    
class FluidSubstitution(MathOps):

    def __init__(self, vp=2525, vs=985, rho=2130, k_min=36.6e9, phi=.3, S_w=.35, rho_w=1070, k_w=3e9, rho_oil=850, k_oil=1.3e9, rho_min=2650):
        super().__init__(vp, vs, rho)
        self.k_min = k_min
        self.rho_min = rho_min
        self.phi = phi
        self.S_w = S_w
        self.rho_w = rho_w
        self.k_w = k_w
        self.rho_oil = rho_oil
        self.k_oil = k_oil

    def calc_moduli(self):
        k_sat = self.rho * (self.vp**2 - (4/3)*self.vs**2)
        mu = self.rho * self.vs**2
        return k_sat, mu
    
    # Compute the effective fluid properties
    def calc_k_fl_init(self):
        k_fl_init = 1 / (self.S_w/self.k_w + (1 - self.S_w)/(self.k_oil))
        rho_fl_init = self.S_w * self.rho_w + (1 - self.S_w) * self.rho_oil
        return k_fl_init, rho_fl_init
    
    # Calculate K dry (no fluids inside)
    def calc_k_dry(self):
        k_sat, mu = self.calc_moduli()
        k_fl_init, rho_fl_init = self.calc_k_fl_init()
        num = k_sat * ((self.phi*self.k_min) / k_fl_init + 1 - self.phi) - self.k_min
        den = (self.phi * self.k_min) / k_fl_init + k_sat / self.k_min - 1 - self.phi 
        return num/den
    
    # Calculate final moduli
    def calc_fin_moduli(self):
        k_fl_fin = self.k_w
        rho_fl_fin = self.rho_w
        return k_fl_fin, rho_fl_fin

    # Calculate the the new-fluid saturated moduli
    def calc_k_sat_fin(self):
        k_dry = self.calc_k_dry()
        k_fl_fin, rho_fl_fin = self.calc_fin_moduli()
        k_sat_fin = k_dry + ( (1 - k_dry / self.k_min)**2 / (self.phi / k_fl_fin + (1 - self.phi) / self.k_min) - k_dry / self.k_min**2)
        return k_sat_fin
    
    # Transform the density
    def calc_rho_fin(self):
        k_fl_fin, rho_fl_fin = self.calc_fin_moduli()
        rho_fin = self.rho_min * (1 - self.phi) + rho_fl_fin * self.phi
        return rho_fin
    
    # Compute the new velocities
    def calc_new_vel(self):
        k_sat, mu = self.calc_moduli()
        rho_fin = self.calc_rho_fin()
        k_sat_fin = self.calc_k_sat_fin()
        vs = np.sqrt(mu / rho_fin)
        vp = np.sqrt( (k_sat_fin + (4/3)*mu) / (rho_fin))
        return vp, vs



class Error:

    def __init__(self, vp, vs, rho, y_true, y_pred):
        super().__init__(self, vp, vs, rho)
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
    
    def remove_outliers(self, df, threshold=3, drop_col=None):
        if drop_col:
            df_copy = df.drop(columns=drop_col)
        z_scores = stats.zscore(df_copy)
        abs_z_scores = abs(z_scores)
        filtered_entries = (abs_z_scores < threshold).all(axis=1)  # Adjust the threshold as needed (e.g., 3 standard deviations)
        df_filtered = df[filtered_entries]
        df_filtered.reset_index(drop=True, inplace=True)
        return df_filtered

class ConversionTool(MathOps):

    def __init__(self, lambda_min_end=1, vp=2500, vs=1500, rho=2100, initial_dept=0, depth=0):
        super().__init__( vp, vs, rho, depth, lambda_min_end,initial_dept)

    def backus_downsampling(self):
        vp0, vs0, rho0, dept0 = self.calc_equivalent_model()
        print(f'Original sampling frequency (in well-log domain) is: {max(self.vp) / (1*0.3048)} Hz')
        print(f'New sampling frequency (in seismic domain) is: {max(vp0) / (self.lambda_min_end*0.3048)} Hz')
        return vp0, vs0, rho0, dept0
    
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