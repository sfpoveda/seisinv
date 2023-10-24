import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

class DensityAnalysis:
    
    def gardner_equation(self, x, a, b):
        
        return a * x**b
    
    def calc_Gardner_relation(self, df):

        for tr in np.unique(df['TR']):

            mask = (df['TR'] == tr)
            df_filtered = df[mask]
            df_filtered.reset_index(drop=True, inplace=True)
            df_filtered.sort_values(by='VP', ignore_index=True, inplace=True)
            # fit the data
            popt, _ = curve_fit(self.gardner_equation, df_filtered['VP'], df_filtered['RHOB'])
            # plot fit
            plt.scatter(df_filtered['VP'], df_filtered['RHOB'], s=1)
            R2 = r2_score(df_filtered['RHOB'], self.gardner_equation(df_filtered['VP'], *popt)) # R2 correlation coefficient
            print(popt)
            plt.plot(df_filtered['VP'], self.gardner_equation(df_filtered['VP'], *popt), 'r', label=f'R2: {np.round(R2, 3)}')

            plt.xlabel(r'$V_p$ $\left(\frac{m}{s}\right)$')
            plt.ylabel(r'$\rho$ $\left(\frac{kg}{m^3}\right)$')
            plt.title(f'Gardner relation for TR: {tr}')
            # Get the x-axis and y-axis limits
            x_limits = plt.xlim()
            y_limits = plt.ylim()
            x_offset, y_offset = x_limits[-1]*.1, y_limits[0]*.025
            plt.legend()
            plt.show()

    def Z_rho_equation(self, x, a, b):
        
        return a * x + b
    
    def calc_Z_rho_relation(self, df):
    
        for tr in np.unique(df['TR']):

            mask = (df['TR'] == tr)
            df_filtered = df[mask]
            df_filtered.reset_index(drop=True, inplace=True)
            df_filtered.sort_values(by='Z', ignore_index=True, inplace=True)
            # fit the data
            popt, _ = curve_fit(self.Z_rho_equation, df_filtered['Z'], df_filtered['RHOB'])
            # plot fit
            plt.scatter(df_filtered['Z'], df_filtered['RHOB'], s=1)
            R2 = r2_score(df_filtered['RHOB'], self.Z_rho_equation(df_filtered['Z'], *popt)) # R2 correlation coefficient
            print(popt)
            plt.plot(df_filtered['Z'], self.Z_rho_equation(df_filtered['Z'], *popt), 'r', label=f'R2: {R2}')

            plt.xlabel(r'$Z$ $\left(\frac{kg}{m^2s}\right)$')
            plt.ylabel(r'$\rho$ $\left(\frac{kg}{m^3}\right)$')
            plt.title(f'No Gardner relation for TR: {tr}')
            # Get the x-axis and y-axis limits
            x_limits = plt.xlim()
            y_limits = plt.ylim()
            x_offset, y_offset = x_limits[-1]*.1, y_limits[0]*.025
            plt.legend()
            plt.show()
            
     
    def Gardner_relation_simple(self, Vp, units='m/s'):
        
        if units == 'm/s':
            rho = 310*Vp**0.25
            
        elif units == 'ft/s':
            rho = 0.23*Vp**0.25
            
        return rho