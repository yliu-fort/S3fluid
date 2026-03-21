import numpy as np
import scipy.special as sp

class NumPySHT:
    """
    Pure NumPy implementation of Spherical Harmonics Transform (SHT).
    Serves as the numerical ground truth and fallback for the Kivy GPGPU simulator.
    """
    def __init__(self, l_max, n_lat=None, n_lon=None):
        self.l_max = l_max
        # By default, use Gauss-Legendre grids with enough resolution
        self.n_lat = n_lat if n_lat is not None else int(np.ceil(1.5 * l_max))
        self.n_lon = n_lon if n_lon is not None else 2 * self.n_lat

        # Ensure minimum size to prevent errors
        if self.n_lat < 2: self.n_lat = 2
        if self.n_lon < 2: self.n_lon = 2

        # 1. Gauss-Legendre Nodes
        x, self.w = sp.roots_legendre(self.n_lat)
        self.lats = np.arccos(x)
        self.lons = np.linspace(0, 2 * np.pi, self.n_lon, endpoint=False)
        self.x = x
        self.sin_theta = np.sin(self.lats)

        # 2. Base Functions precomputations
        self.P_lm = np.zeros((self.l_max, self.l_max, self.n_lat))
        self.dP_lm = np.zeros((self.l_max, self.l_max, self.n_lat))

        for l in range(self.l_max):
            for m in range(l + 1):
                y, dy = sp.sph_harm_y(l, m, self.lats, 0.0, diff_n=1)
                self.P_lm[l, m, :] = y.real
                self.dP_lm[l, m, :] = dy[:, 0].real

    def forward_sht(self, data):
        """
        Transforms physical grid data (lat, lon) into spectral coefficients (l, m).
        Args:
            data: ndarray of shape (n_lat, n_lon)
        Returns:
            f_lm: ndarray of shape (l_max, l_max) dtype=complex
        """
        if data.shape != (self.n_lat, self.n_lon):
            raise ValueError(f"Input shape {data.shape} does not match grid {(self.n_lat, self.n_lon)}")

        f_m = np.fft.rfft(data, axis=-1)
        f_lm = np.zeros((self.l_max, self.l_max), dtype=complex)

        # Integrate over theta with Gauss-Legendre quadrature
        # We also need to multiply by 2*pi/n_lon for the phi integration part
        factor = 2.0 * np.pi / self.n_lon

        for l in range(self.l_max):
            for m in range(l + 1):
                if m < f_m.shape[1]:
                    # Using einsum or tensordot for speed
                    integral = factor * np.sum(self.w * f_m[:, m] * self.P_lm[l, m, :])
                    f_lm[l, m] = integral

        return f_lm

    def inverse_sht(self, f_lm):
        """
        Transforms spectral coefficients (l, m) into physical grid data (lat, lon).
        Args:
            f_lm: ndarray of shape (l_max, l_max) dtype=complex
        Returns:
            data: ndarray of shape (n_lat, n_lon) dtype=float
        """
        if f_lm.shape != (self.l_max, self.l_max):
            raise ValueError(f"Input shape {f_lm.shape} does not match (l_max, l_max) {(self.l_max, self.l_max)}")

        f_m = np.zeros((self.n_lat, self.n_lon // 2 + 1), dtype=complex)

        for l in range(self.l_max):
            for m in range(l + 1):
                if m < f_m.shape[1]:
                    f_m[:, m] += f_lm[l, m] * self.P_lm[l, m, :]

        f_m_scaled = f_m * self.n_lon
        data = np.fft.irfft(f_m_scaled, n=self.n_lon, axis=-1)
        return data

    def inverse_sht_grad(self, f_lm):
        """
        Computes the gradient (d/dphi, d/dtheta) of the field described by f_lm directly from spectral space.
        Args:
            f_lm: ndarray of shape (l_max, l_max) dtype=complex
        Returns:
            grad_phi: ndarray (n_lat, n_lon)
            grad_th: ndarray (n_lat, n_lon)
        """
        if f_lm.shape != (self.l_max, self.l_max):
            raise ValueError(f"Input shape {f_lm.shape} does not match (l_max, l_max) {(self.l_max, self.l_max)}")

        f_m_dphi = np.zeros((self.n_lat, self.n_lon // 2 + 1), dtype=complex)
        f_m_dth = np.zeros((self.n_lat, self.n_lon // 2 + 1), dtype=complex)

        for l in range(self.l_max):
            for m in range(l + 1):
                if m < f_m_dphi.shape[1]:
                    f_m_dphi[:, m] += f_lm[l, m] * (1j * m) * self.P_lm[l, m, :]
                    f_m_dth[:, m] += f_lm[l, m] * self.dP_lm[l, m, :]

        grad_phi = np.fft.irfft(f_m_dphi * self.n_lon, n=self.n_lon, axis=-1)
        grad_th = np.fft.irfft(f_m_dth * self.n_lon, n=self.n_lon, axis=-1)

        return grad_phi, grad_th
