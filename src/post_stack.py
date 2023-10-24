import sys
sys.path.append('D:\\Users\\serfe\\Desktop\\Maestria geofísica\\1- Semestre 202302\\Inversión sísmica\\999 - Source')
from basic_ops_seis_inv import MathOps
from basic_ops_seis_inv import Error
import numpy as np

math_ops = MathOps()
error = Error()

class poststack_methods:
    
    def recursive_inversion(self, df, method):
    
        temp = []
        z0 = df['Z'].iloc[0]

        if method == 'r any':
            cociente = 1
            # calculate the Acoustic Impedance (Z)
            for i in range(len(df['r_prime'])-1):
                cociente *= (1+df['r_prime'][i])/(1-df['r_prime'][i])
                temp.append(cociente)
            temp.append(temp[-1])
            temp = np.array(temp)*z0

        if method == 'r < 0.3':
            # calculate Z
            for i in range(len(df['r_prime'])-1):
                z_temp = z0*np.exp(2*np.sum(df['r_prime'][:i+1]))
                temp.append(z_temp)
            temp.append(temp[-1])

        if method == 'r < 0.1':
            # calculate Z
            for i in range(len(df['r_prime'])-1):
                z_temp = (2*z0)*np.sum(df['r_prime'][:i+1])
                temp.append(z_temp)
            temp.append(temp[-1])

        df[f'Z_{method}'] = temp

        return df
    
    def MBI(self, m0, r0, r, step=0.01, max_iter=20, regularization_coeff=0.1, early_stop_error=.01):
        
        # Initialize variables and setup
        for i in range(max_iter):
            
            # Calculate Jacobian matrix G
            G = math_ops.calc_G(m0, r0, r, step=step)
            # Compute regularization term
            regularization_term = regularization_coeff * np.eye(len(m0))
            # Update impedance model m0 using the regularized inverse
            m0 -= np.linalg.inv(G.T @ G + regularization_term) @ G.T @ (r - r0)
            # Update r0 based on the updated m0
            r0 = math_ops.calc_r(m0)
            # Calculate and print errors
            L1 = error.L1_error(r, r0)
            L2 = error.L2_error(r, r0)
            print(f'Iteration {i + 1}: L1 error={L1}, L2 error={L2}')
            # Check for convergence based on error thresholds
            if L1 < early_stop_error and L2 < early_stop_error:
                break

        # Print results
        print(f'Converged after {i + 1} iterations')
        print(f'Final L1 error: {L1}')
        print(f'Final L2 error: {L2}')

        return m0, r0

