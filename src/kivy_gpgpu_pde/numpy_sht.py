import numpy as np
import math
from scipy.special import roots_legendre, lpmv

class NumPySHT:
    def __init__(self, l_max, n_lat=None, n_lon=None):
        self.l_max = l_max
        self.n_lat = n_lat if n_lat is not None else int(1.5 * l_max)
        self.n_lon = n_lon if n_lon is not None else 2 * self.n_lat

        self.lons = np.linspace(0, 2 * np.pi, self.n_lon, endpoint=False)

        x, self.weights = roots_legendre(self.n_lat)
        self.lats = np.arccos(x)
        self.sin_theta = np.sin(self.lats)
        self.cos_theta = x

        self.m_max = self.l_max
        self.m_values = np.fft.rfftfreq(self.n_lon, d=1.0/self.n_lon).astype(int)
        self.m_values = self.m_values[self.m_values <= self.m_max]
        self.num_m = len(self.m_values)

        self.alp = np.zeros((self.num_m, self.l_max + 1, self.n_lat))
        self.dalp_dtheta = np.zeros_like(self.alp)

        self._precompute_alp()

    def _precompute_alp(self):
        for i_m, m in enumerate(self.m_values):
            for l in range(m, self.l_max + 1):
                p_lm_unnorm = lpmv(m, l, self.cos_theta)

                N_lm = np.sqrt( (2*l+1)/(4*np.pi) * math.factorial(l-m) / math.factorial(l+m) )
                phase = (-1)**m

                p_lm_norm = phase * N_lm * p_lm_unnorm
                self.alp[i_m, l, :] = p_lm_norm

                if l == m:
                    p_l_minus_1_m_unnorm = np.zeros_like(self.cos_theta)
                else:
                    p_l_minus_1_m_unnorm = lpmv(m, l-1, self.cos_theta)

                sin_t = np.where(self.sin_theta == 0, 1e-15, self.sin_theta)

                dp_dtheta_unnorm = (l * self.cos_theta * p_lm_unnorm - (l + m) * p_l_minus_1_m_unnorm) / sin_t
                self.dalp_dtheta[i_m, l, :] = phase * N_lm * dp_dtheta_unnorm

    def forward_sht(self, data):
        data_m = np.fft.rfft(data, axis=-1) * (2 * np.pi / self.n_lon)
        data_m = data_m[:, :self.num_m]

        coeffs = np.zeros((self.num_m, self.l_max + 1), dtype=complex)

        for i_m, m in enumerate(self.m_values):
            f_m = data_m[:, i_m]
            p_lm = self.alp[i_m, :, :]
            integrand = f_m * self.weights
            coeffs[i_m, m:] = np.dot(p_lm[m:, :], integrand)

        return coeffs

    def inverse_sht(self, coeffs):
        data_m = np.zeros((self.n_lat, self.n_lon // 2 + 1), dtype=complex)
        for i_m, m in enumerate(self.m_values):
            f_lm = coeffs[i_m, :]
            p_lm_t = self.alp[i_m, :, :].T
            data_m[:, i_m] = np.dot(p_lm_t[:, m:], f_lm[m:])

        data_m = data_m * self.n_lon
        data = np.fft.irfft(data_m, n=self.n_lon, axis=-1)
        return data

    def inverse_sht_grad(self, coeffs):
        d_data_dtheta_m = np.zeros((self.n_lat, self.n_lon // 2 + 1), dtype=complex)
        d_data_dphi_m = np.zeros_like(d_data_dtheta_m)

        for i_m, m in enumerate(self.m_values):
            f_lm = coeffs[i_m, :]

            dalp_dtheta_t = self.dalp_dtheta[i_m, :, :].T
            d_data_dtheta_m[:, i_m] = np.dot(dalp_dtheta_t[:, m:], f_lm[m:])

            p_lm_t = self.alp[i_m, :, :].T
            f_m = np.dot(p_lm_t[:, m:], f_lm[m:])
            d_data_dphi_m[:, i_m] = f_m * (1j * m)

        d_data_dtheta_m = d_data_dtheta_m * self.n_lon
        d_data_dphi_m = d_data_dphi_m * self.n_lon

        d_data_dtheta = np.fft.irfft(d_data_dtheta_m, n=self.n_lon, axis=-1)
        d_data_dphi = np.fft.irfft(d_data_dphi_m, n=self.n_lon, axis=-1)

        return d_data_dphi, d_data_dtheta

    def synthesize_vector(self, u_coeffs, v_coeffs):
        pass
