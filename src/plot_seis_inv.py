import matplotlib.pyplot as plt
import numpy as np

class PlotSeisInv:

    def plot_results(self, df, method, r, dict_names):
        fig, ax = plt.subplots(1,5, figsize=(10,7), sharey=True, dpi=500)
        lw=.3
        i = 0
        if method:
            key = f'DT_{method}'
        else:
            key = 'DT'
        ax[i].plot(df[key], df['DEPT'], linewidth=lw)
        ax[i].set_title(r'$DT$ $\left(\frac{s}{m}\right)$')
        ax[i].set_ylabel(r'Depth ($m$)')
        ax[i].grid(alpha=.5)
        ax[i].invert_yaxis()
        ax[i].set_axisbelow(True)
        ax[i].ticklabel_format(style='sci', axis='x', scilimits=(-4,-4))
        ax[i].tick_params(axis='x', labelsize=6)
        i = 1
        if method:
            key = f'VP_{method}'
        else:
            key = 'VP'
        ax[i].plot(df[key], df['DEPT'], linewidth=lw)
        ax[i].set_title(r'$VP$ $\left(\frac{m}{s}\right)$')
        ax[i].grid(alpha=.5)
        ax[i].invert_yaxis()
        ax[i].set_axisbelow(True)
        ax[i].ticklabel_format(style='sci', axis='x', scilimits=(3,3))
        ax[i].tick_params(axis='x', labelsize=6)
        i = 2
        if method:
            key = f'RHOB_{method}'
        else:
            key='RHOB'
        ax[i].plot(df[key], df['DEPT'], linewidth=lw)
        ax[i].set_title(r'$\rho$ $\left(\frac{kg}{m^3}\right)$')
        ax[i].grid(alpha=.5)
        ax[i].set_axisbelow(True)
        ax[i].ticklabel_format(style='sci', axis='x')
        ax[i].ticklabel_format(style='sci', axis='x', scilimits=(3,3))
        ax[i].tick_params(axis='x', labelsize=6)
        i = 3
        if method:
            key = f'Z_{method}'
        else:
            key='Z'
        ax[i].plot(df[key], df['DEPT'], linewidth=lw)
        ax[i].set_title(r'$Z$ $\left(\frac{kg}{s \cdot m^2}\right)$')
        ax[i].grid(alpha=.5)
        ax[i].set_axisbelow(True) 
        ax[i].ticklabel_format(style='sci', axis='x', scilimits=(6,6))
        ax[i].tick_params(axis='x', labelsize=6)
        i = 4
        key = r
        ax[i].plot(df[key], df['DEPT'], linewidth=lw)
        ax[i].set_title(r'$r$')
        ax[i].grid(alpha=.5)
        ax[i].set_axisbelow(True)
        ax[i].ticklabel_format(style='sci', axis='x')
        ax[i].tick_params(axis='x', labelsize=6)
        '''i = 5
        key = f'waveform_{r}'
        ax[i].plot(df[key], df['DEPT'], linewidth=.1, color='k')
        ax[i].set_title(r'$A$')
        ax[i].grid(alpha=.5)
        ax[i].set_axisbelow(True)
        ax[i].fill_betweenx(df['DEPT'], x1=df[f'waveform_{r}'], x2=0, where=df[f'waveform_{r}']<=0, color='blue', linewidth=0)
        ax[i].fill_betweenx(df['DEPT'], x1=df[f'waveform_{r}'], x2=0, where=df[f'waveform_{r}']>0, color='red', linewidth=0)
        ax[i].ticklabel_format(style='sci', axis='x')
        ax[i].tick_params(axis='x', labelsize=6)

        i = 6
        key = 'TR'
        lithology_data = df['TR']
        unique_labels = np.unique(df['TR'])
        colors = {-9999:'saddlebrown', 1:'sandybrown', 2:'darkgoldenrod', 3:'brown', 4:'dimgray', 5:'yellow'}
        # Plot lithology data
        for j, lithology in enumerate(unique_labels):
            mask = (lithology_data==lithology)
            ax[i].fill_betweenx(df['DEPT'], x1=0, x2=1, where=mask, label=f'{dict_names[unique_labels[j]]}', color=colors[lithology])
        ax[i].set_title(r'$TR$')
        ax[i].grid(alpha=.5)
        ax[i].set_axisbelow(True)
        ax[i].legend(loc='upper right', fontsize=4)
        ax[i].tick_params(axis='x', labelsize=6)'''
        plt.suptitle(f'Method {method}')