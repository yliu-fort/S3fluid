import numpy as np
from pde_simulator.numpy_sht import NumPySHT

def test_initialization():
    lmax = 3
    N_lat = 16
    N_lon = 32
    sht = NumPySHT(lmax, N_lat, N_lon)

    assert sht.lats.shape == (N_lat,)
    assert sht.lons.shape == (N_lon,)
    assert sht.P.shape == (lmax + 1, lmax + 1, N_lat)
    assert sht.dP.shape == (lmax + 1, lmax + 1, N_lat)

def test_forward_inverse_sht():
    sht = NumPySHT(lmax=3, N_lat=16, N_lon=32)

    # Create simple data: f(theta, phi) = cos(theta) + sin(theta)cos(phi)
    data = np.cos(sht.lats)[:, None] + np.sin(sht.lats)[:, None] * np.cos(sht.lons)[None, :]

    coeffs = sht.forward_sht(data)
    data_rec = sht.inverse_sht(coeffs)

    error = np.max(np.abs(data - data_rec))
    assert error < 1e-12, f"Reconstruction error too large: {error}"

def test_inverse_sht_grad():
    sht = NumPySHT(lmax=3, N_lat=16, N_lon=32)

    # f(theta, phi) = sin(theta) cos(phi)
    data = np.sin(sht.lats)[:, None] * np.cos(sht.lons)[None, :]
    coeffs = sht.forward_sht(data)

    grad_lat, grad_lon = sht.inverse_sht_grad(coeffs)

    anal_grad_lat = np.cos(sht.lats)[:, None] * np.cos(sht.lons)[None, :]
    anal_grad_lon = -np.sin(sht.lons)[None, :]

    err_lat = np.max(np.abs(grad_lat - anal_grad_lat))
    err_lon = np.max(np.abs(grad_lon - anal_grad_lon))

    assert err_lat < 1e-10, f"Latitudinal gradient error too large: {err_lat}"
    assert err_lon < 1e-10, f"Longitudinal gradient error too large: {err_lon}"
