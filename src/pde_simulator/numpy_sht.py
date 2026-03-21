import numpy as np
import scipy.special as sp

class NumPySHT:
    """
    NumPy-based Spherical Harmonics Transform (SHT) engine.
    Used for ground truth validation and core math logic.
    Provides forward and inverse SHT, as well as gradient computations.
    """
    def __init__(self, lmax: int, N_lat: int, N_lon: int):
        self.lmax = lmax
        self.N_lat = N_lat
        self.N_lon = N_lon

        # 1. Gauss-Legendre quadrature nodes and weights
        self.x, self.w = sp.roots_legendre(N_lat)
        self.lats = np.arccos(self.x)  # colatitudes (0 to pi)
        self.lons = np.linspace(0, 2 * np.pi, N_lon, endpoint=False) # longitudes (0 to 2pi)

        # 2. Precompute Associated Legendre Polynomials (ALP) and derivatives
        # Shape: (lmax+1, lmax+1, N_lat)
        self.P = np.zeros((lmax + 1, lmax + 1, N_lat))
        self.dP = np.zeros((lmax + 1, lmax + 1, N_lat))

        for l in range(lmax + 1):
            for m in range(l + 1):
                y = sp.sph_harm_y(l, m, self.lats, 0.0)
                self.P[l, m, :] = y.real

                # Derivative w.r.t colatitude (theta)
                eps = 1e-5
                y_plus = sp.sph_harm_y(l, m, self.lats + eps, 0.0)
                y_minus = sp.sph_harm_y(l, m, self.lats - eps, 0.0)
                self.dP[l, m, :] = (y_plus.real - y_minus.real) / (2 * eps)

    def forward_sht(self, data: np.ndarray) -> np.ndarray:
        """
        Computes forward Spherical Harmonics Transform.
        Input data shape: (N_lat, N_lon)
        Returns: Complex coefficients shape (lmax+1, lmax+1)
        """
        # Fourier transform along longitude
        f_m = np.fft.rfft(data, axis=1) / self.N_lon

        # We only need m up to lmax
        m_max = min(self.lmax + 1, f_m.shape[1])
        f_m_trunc = f_m[:, :m_max]
        f_m_w_trunc = f_m_trunc * self.w[:, None]

        coeffs = np.zeros((self.lmax + 1, self.lmax + 1), dtype=complex)

        # Legendre transform using np.einsum for vectorization
        # P shape: (lmax+1, lmax+1, N_lat)
        # f_m_w_trunc shape: (N_lat, m_max)
        # We compute sum over k (latitudes).
        # We need to slice P to match m_max.
        coeffs[:, :m_max] = 2 * np.pi * np.einsum('lmk,km->lm', self.P[:, :m_max, :], f_m_w_trunc)

        return coeffs

    def inverse_sht(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Computes inverse Spherical Harmonics Transform.
        Input coeffs shape: (lmax+1, lmax+1)
        Returns: Real spatial data shape (N_lat, N_lon)
        """
        f_m = np.zeros((self.N_lat, self.N_lon // 2 + 1), dtype=complex)

        m_max = min(self.lmax + 1, f_m.shape[1])

        # f_m_trunc shape: (N_lat, m_max)
        # coeffs shape: (lmax+1, m_max)
        # P shape: (lmax+1, m_max, N_lat)
        f_m_trunc = np.einsum('lm,lmk->km', coeffs[:, :m_max], self.P[:, :m_max, :])
        f_m[:, :m_max] = f_m_trunc

        return np.fft.irfft(f_m * self.N_lon, n=self.N_lon, axis=1)

    def inverse_sht_grad(self, coeffs: np.ndarray):
        """
        Computes the gradient of a scalar field from its spectral coefficients.
        Returns:
            grad_lat: d f / d theta (shape: N_lat, N_lon)
            grad_lon: (1/sin theta) d f / d phi (shape: N_lat, N_lon)
        """
        f_m_lat = np.zeros((self.N_lat, self.N_lon // 2 + 1), dtype=complex)
        f_m_lon = np.zeros((self.N_lat, self.N_lon // 2 + 1), dtype=complex)

        sin_lats = np.sin(self.lats)
        inv_sin = np.zeros_like(sin_lats)
        mask = sin_lats > 1e-10
        inv_sin[mask] = 1.0 / sin_lats[mask]

        m_max = min(self.lmax + 1, f_m_lat.shape[1])

        # Latitudinal gradient (d/d theta)
        f_m_lat_trunc = np.einsum('lm,lmk->km', coeffs[:, :m_max], self.dP[:, :m_max, :])
        f_m_lat[:, :m_max] = f_m_lat_trunc

        # Longitudinal gradient ( (1/sin theta) d/d phi )
        m_arr = np.arange(m_max)
        coeffs_lon = coeffs[:, :m_max] * (1j * m_arr)[None, :]
        f_m_lon_trunc = np.einsum('lm,lmk->km', coeffs_lon, self.P[:, :m_max, :]) * inv_sin[:, None]
        f_m_lon[:, :m_max] = f_m_lon_trunc

        grad_lat = np.fft.irfft(f_m_lat * self.N_lon, n=self.N_lon, axis=1)
        grad_lon = np.fft.irfft(f_m_lon * self.N_lon, n=self.N_lon, axis=1)
        return grad_lat, grad_lon
