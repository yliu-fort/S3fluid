import numpy as np

def noise_psd1(N, psd = lambda f: 1):
        X_white = np.fft.rfft(np.random.randn(N))
        S = psd(np.fft.rfftfreq(N))
        # Normalize S
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S
        return np.fft.irfft(X_shaped)

def PSDGenerator1(f):
    return lambda N: noise_psd1(N, f)

@PSDGenerator1
def white_noise1(f):
    return 1

@PSDGenerator1
def blue_noise1(f):
    return np.sqrt(f)

@PSDGenerator1
def violet_noise1(f):
    return f

@PSDGenerator1
def brownian_noise1(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator1
def pink_noise1(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

def add_arrays_to_ndarray(ndarray, *arrays):
    ndims = len(ndarray.shape)
    for i, arr in enumerate(arrays):
        for _ in range(i):
            arr = np.expand_dims(arr, axis=0)
        for _ in range(ndims-i-1):
            arr = np.expand_dims(arr, axis=-1)
        ndarray = ndarray + arr
    return ndarray

def noise_psd(N, psd = lambda f: 1):
    ndims = len(N)
    X_white = np.fft.rfftn(np.random.randn(*N))
    fsqr_arrays = tuple(np.fft.fftfreq(N[idim])**2 for idim in range(ndims-1)) + (np.fft.rfftfreq(N[-1])**2,)
    f_radii = np.zeros(X_white.shape)
    f_radii = np.sqrt(add_arrays_to_ndarray(f_radii, *fsqr_arrays))
    S = psd(f_radii)
    # Normalize S
    S = S / np.sqrt(np.mean(S**2))
    X_shaped = X_white * S
    return np.fft.irfftn(X_shaped)

def PSDGenerator(f):
    return lambda *N: noise_psd(N, f)

@PSDGenerator
def white_noise(f):
    return 1

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)

@PSDGenerator
def violet_noise(f):
    return f

@PSDGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    def plot_spectrum(s):
        f = np.fft.rfftfreq(len(s))
        return plt.loglog(f, np.abs(np.fft.rfft(s)))[0]
    
    plt.style.use('dark_background')
    plt.figure(figsize=(8, 6), tight_layout=True)
    for G, c in zip(
            [brownian_noise, pink_noise, white_noise, blue_noise, violet_noise], 
            ['brown', 'hotpink', 'white', 'blue', 'violet']):
        plot_spectrum(G(30*50_000).flatten()).set(color=c, linewidth=3)
    plt.legend(['brownian', 'pink', 'white', 'blue', 'violet'])
    plt.suptitle("Colored Noise")
    plt.ylim([1e-3, None])
    
    for G, c in zip(
        [brownian_noise, pink_noise, white_noise, blue_noise, violet_noise], 
        ['brown', 'hotpink', 'white', 'blue', 'violet']):
        plt.figure(figsize=(4, 3), tight_layout=True)
        plt.pcolor(G(1024, 1024), cmap='viridis')
        plt.colorbar(label='Values')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(c)
    plt.show()