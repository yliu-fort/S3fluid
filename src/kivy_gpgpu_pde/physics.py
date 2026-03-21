import numpy as np

class NavierStokesRHS:
    """
    Physical module for the Navier-Stokes equations on a sphere
    in vorticity-streamfunction formulation.
    """
    def __init__(self, sht, nu=0.0):
        self.sht = sht
        self.nu = nu

        # Precompute laplacian eigenvalues: -l(l+1)
        self.laplacian = np.zeros((sht.num_m, sht.l_max + 1))
        self.inv_laplacian = np.zeros((sht.num_m, sht.l_max + 1))

        for i_m, m in enumerate(sht.m_values):
            for l in range(m, sht.l_max + 1):
                self.laplacian[i_m, l] = -float(l * (l + 1))
                if l > 0:
                    self.inv_laplacian[i_m, l] = -1.0 / float(l * (l + 1))

    def numpy_nonl(self, zeta_coeffs):
        """
        Pure NumPy RHS for testing and baseline (UT-08).
        zeta_coeffs: spectral coefficients of vorticity (num_m, l_max + 1) complex array
        Returns: d(zeta_coeffs)/dt complex array
        """
        # 1. Compute streamfunction psi
        psi_coeffs = zeta_coeffs * self.inv_laplacian

        # 2. Compute gradients in physical space
        dpsi_dphi, dpsi_dtheta = self.sht.inverse_sht_grad(psi_coeffs)
        dzeta_dphi, dzeta_dtheta = self.sht.inverse_sht_grad(zeta_coeffs)

        # 3. Compute Jacobian J(psi, zeta)
        # J = (dpsi_dphi * dzeta_dtheta - dpsi_dtheta * dzeta_dphi) / sin(theta)
        sin_theta = self.sht.sin_theta[:, np.newaxis]
        # Avoid division by zero at poles
        sin_theta = np.where(sin_theta == 0, 1e-15, sin_theta)

        J = (dpsi_dphi * dzeta_dtheta - dpsi_dtheta * dzeta_dphi) / sin_theta

        # 4. Transform Jacobian back to spectral space
        J_coeffs = self.sht.forward_sht(J)

        # 5. Calculate diffusion term in spectral space
        diffusion_coeffs = self.nu * self.laplacian * zeta_coeffs

        # d(zeta_lm)/dt = -J_lm + nu * nabla^2 zeta_lm
        rhs_coeffs = -J_coeffs + diffusion_coeffs

        return rhs_coeffs
