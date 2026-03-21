import numpy as np
import scipy.special as sp
from src.kivy_gpgpu_pde.numpy_sht import NumPySHT

def test_initialization():
    sht = NumPySHT(l_max=4, n_lat=8, n_lon=16)
    assert sht.l_max == 4
    assert sht.n_lat == 8
    assert sht.n_lon == 16
    assert sht.P_lm.shape == (4, 4, 8)
    assert sht.dP_lm.shape == (4, 4, 8)

def test_forward_inverse_sht_Y20():
    sht = NumPySHT(l_max=4, n_lat=8, n_lon=16)
    L, M = np.meshgrid(sht.lons, sht.lats)

    # Y_2^0
    data = sp.sph_harm_y(2, 0, sht.lats[:, None], L).real

    # Forward SHT
    flm = sht.forward_sht(data)

    # Check coefficients
    # Should be 1.0 at l=2, m=0 and near 0 elsewhere
    assert np.isclose(flm[2, 0].real, 1.0, atol=1e-10)
    assert np.isclose(flm[0, 0].real, 0.0, atol=1e-10)

    # Inverse SHT
    rec_data = sht.inverse_sht(flm)

    # Check reconstruction
    assert np.allclose(data, rec_data, atol=1e-10)

def test_forward_inverse_sht_Y31():
    sht = NumPySHT(l_max=4, n_lat=8, n_lon=16)
    L, M = np.meshgrid(sht.lons, sht.lats)

    # 2 * Re(Y_3^1)
    data = 2.0 * sp.sph_harm_y(3, 1, sht.lats[:, None], L).real

    # Forward
    flm = sht.forward_sht(data)

    assert np.isclose(flm[3, 1].real, 1.0, atol=1e-10)

    # Inverse
    rec_data = sht.inverse_sht(flm)

    assert np.allclose(data, rec_data, atol=1e-10)

def test_inverse_sht_grad():
    sht = NumPySHT(l_max=4, n_lat=8, n_lon=16)
    L, M = np.meshgrid(sht.lons, sht.lats)

    # 2 * Re(Y_3^1)
    data = 2.0 * sp.sph_harm_y(3, 1, sht.lats[:, None], L).real
    flm = sht.forward_sht(data)

    grad_phi, grad_th = sht.inverse_sht_grad(flm)

    # analytic d/dphi
    test_dphi = -2.0 * 1 * sp.sph_harm_y(3, 1, sht.lats[:, None], L).imag
    assert np.allclose(grad_phi, test_dphi, atol=1e-10)

    # analytic d/dtheta
    y, dy = sp.sph_harm_y(3, 1, sht.lats[:, None], L, diff_n=1)
    test_dth = 2.0 * dy[:,:,0].real

    assert np.allclose(grad_th, test_dth, atol=1e-10)

def test_Ylm_scaling():
    sht = NumPySHT(l_max=4, n_lat=8, n_lon=16)
    L, M = np.meshgrid(sht.lons, sht.lats)

    for l in range(4):
        for m in range(l + 1):
            if m == 0:
                data = sp.sph_harm_y(l, m, sht.lats[:, None], L).real
                flm = sht.forward_sht(data)
                assert np.isclose(flm[l, m].real, 1.0, atol=1e-10)
            else:
                data = 2.0 * sp.sph_harm_y(l, m, sht.lats[:, None], L).real
                flm = sht.forward_sht(data)
                assert np.isclose(flm[l, m].real, 1.0, atol=1e-10)

                # Check reconstruction
                rec_data = sht.inverse_sht(flm)
                assert np.allclose(data, rec_data, atol=1e-10)

def test_inverse_sht_grad_all():
    sht = NumPySHT(l_max=4, n_lat=8, n_lon=16)
    L, M = np.meshgrid(sht.lons, sht.lats)

    for l in range(4):
        for m in range(l + 1):
            if m == 0:
                data = sp.sph_harm_y(l, m, sht.lats[:, None], L).real
            else:
                data = 2.0 * sp.sph_harm_y(l, m, sht.lats[:, None], L).real

            flm = sht.forward_sht(data)
            grad_phi, grad_th = sht.inverse_sht_grad(flm)

            # analytic d/dphi
            if m == 0:
                test_dphi = np.zeros_like(data)
            else:
                test_dphi = -2.0 * m * sp.sph_harm_y(l, m, sht.lats[:, None], L).imag

            assert np.allclose(grad_phi, test_dphi, atol=1e-10)

            # analytic d/dtheta
            y, dy = sp.sph_harm_y(l, m, sht.lats[:, None], L, diff_n=1)
            if m == 0:
                test_dth = dy[:,:,0].real
            else:
                test_dth = 2.0 * dy[:,:,0].real

            assert np.allclose(grad_th, test_dth, atol=1e-10)
