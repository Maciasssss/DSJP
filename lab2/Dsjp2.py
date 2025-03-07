import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift
from scipy.signal.windows import hann, flattop

# 1. Parameters from variant
f1, f2, f3 = 500, 500.25, 499.75  
A_max = 3  
fs = 800  
N = 1800  

# 2. Generating Signals
k = np.arange(N)
x1 = A_max * np.sin(2 * np.pi * f1/fs * k)
x2 = A_max * np.sin(2 * np.pi * f2/fs * k)
x3 = A_max * np.sin(2 * np.pi * f3/fs * k)

# 3. Generating Windows 
wrect = np.ones(N)
whann = hann(N, sym=False)
wflattop = flattop(N, sym=False)

# Plot the windows
plt.figure(figsize=(10, 6))
plt.plot(wrect, 'C0o-', ms=3, label='rect')
plt.plot(whann, 'C1o-', ms=3, label='hann')
plt.plot(wflattop, 'C2o-', ms=3, label='flattop')
plt.xlabel(r'$k$')
plt.ylabel(r'window $w[k]$')
plt.xlim(0, N)
plt.legend()
plt.grid(True)
plt.title('Window Functions')
plt.savefig("window_functions.png")

# 4. DFT spectra using FFT algorithm
X1wrect = fft(x1)
X2wrect = fft(x2)
X3wrect = fft(x3)

X1whann = fft(x1 * whann)
X2whann = fft(x2 * whann)
X3whann = fft(x3 * whann)

X1wflattop = fft(x1 * wflattop)
X2wflattop = fft(x2 * wflattop)
X3wflattop = fft(x3 * wflattop)

# 4.1 Normalized level of DFT - similar to the example's fft2db function
def fft2db(X):
    N = X.size
    Xtmp = 2/N * X 
    Xtmp[0] *= 1/2  
    
    if N % 2 == 0:  
       
        Xtmp[N//2] = Xtmp[N//2] / 2
    
    return 20 * np.log10(np.abs(Xtmp))  

# Set up frequency vector
df = fs/N
f = np.arange(N) * df

# Plot the normalized DFT spectra
plt.figure(figsize=(16/1.5, 10/1.5))

# Rectangular window
plt.subplot(3, 1, 1)
plt.plot(f, fft2db(X1wrect), 'C0o-', ms=3, label=f'best case rect (f1={f1}Hz)')
plt.plot(f, fft2db(X2wrect), 'C3o-', ms=3, label=f'worst case rect (f2={f2}Hz)')
plt.xlim(f1-25, f1+25) 
plt.ylim(-60, 0)
plt.xticks(np.arange(f1-25, f1+30, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)
plt.title('Rectangular Window DFT')

# Hann window
plt.subplot(3, 1, 2)
plt.plot(f, fft2db(X1whann), 'C0o-', ms=3, label=f'best case hann (f1={f1}Hz)')
plt.plot(f, fft2db(X2whann), 'C3o-', ms=3, label=f'worst case hann (f2={f2}Hz)')
plt.xlim(f1-25, f1+25)
plt.ylim(-60, 0)
plt.xticks(np.arange(f1-25, f1+30, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)
plt.title('Hann Window DFT')

# Flattop window
plt.subplot(3, 1, 3)
plt.plot(f, fft2db(X1wflattop), 'C0o-', ms=3, label=f'best case flattop (f1={f1}Hz)')
plt.plot(f, fft2db(X2wflattop), 'C3o-', ms=3, label=f'worst case flattop (f2={f2}Hz)')
plt.xlim(f1-25, f1+25)
plt.ylim(-60, 0)
plt.xticks(np.arange(f1-25, f1+30, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.xlabel('f / Hz')
plt.ylabel('A / dB')
plt.grid(True)
plt.title('Flattop Window DFT')

plt.tight_layout()
plt.savefig("dft_spectra.png")

# 4.2 Window DTFT spectra - using the example's winDTFTdB approach
def winDTFTdB(w):
    N = w.size 
    Nz = 100 * N  
    W = np.zeros(Nz)  
    W[0:N] = w  
    W = np.abs(fftshift(fft(W)))  
    W /= np.max(W)  
    W = 20 * np.log10(W)  
    
    # get appropriate digital frequencies
    Omega = 2 * np.pi / Nz * np.arange(Nz) - np.pi  
    return Omega, W

# Plot window DTFT spectra
plt.figure(figsize=(12, 8))

# Add reference lines for mainlobe and sidelobe levels - these values are approximations 
plt.plot([-np.pi, +np.pi], [-3.01, -3.01], 'gray', label='mainlobe bandwidth')
plt.plot([-np.pi, +np.pi], [-13.3, -13.3], 'gray', label='rect max sidelobe')
plt.plot([-np.pi, +np.pi], [-31.5, -31.5], 'gray', label='hann max sidelobe')
plt.plot([-np.pi, +np.pi], [-93.6, -93.6], 'gray', label='flattop max sidelobe')

# Plot actual window DTFTs
Omega, W = winDTFTdB(wrect)
plt.plot(Omega, W, label='rect')

Omega, W = winDTFTdB(whann)
plt.plot(Omega, W, label='hann')

Omega, W = winDTFTdB(wflattop)
plt.plot(Omega, W, label='flattop')

plt.xlim(-np.pi, np.pi)
plt.ylim(-120, 10)
plt.xlabel(r'$\Omega$')
plt.ylabel(r'|W($\Omega$)| / dB')
plt.legend()
plt.grid(True)
plt.title('Window DTFT Spectra (Full Range)')
plt.savefig("window_dtft_full.png")

# Zoom in to mainlobe
plt.figure(figsize=(12, 8))
plt.plot([-np.pi, +np.pi], [-3.01, -3.01], 'gray', label='mainlobe bandwidth')

Omega, W = winDTFTdB(wrect)
plt.plot(Omega, W, label='rect')

Omega, W = winDTFTdB(whann)
plt.plot(Omega, W, label='hann')

Omega, W = winDTFTdB(wflattop)
plt.plot(Omega, W, label='flattop')

plt.xlim(-np.pi/100, np.pi/100) 
plt.ylim(-30, 5)
plt.xlabel(r'$\Omega$')
plt.ylabel(r'|W($\Omega$)| / dB')
plt.legend()
plt.grid(True)
plt.title('Window DTFT Spectra (Mainlobe Detail)')
plt.savefig("window_dtft_mainlobe.png")

# Additional plot with the third frequency
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(f, fft2db(X1wrect), 'C0-', label=f'f1={f1}Hz')
plt.plot(f, fft2db(X2wrect), 'C1-', label=f'f2={f2}Hz')
plt.plot(f, fft2db(X3wrect), 'C2-', label=f'f3={f3}Hz')
plt.xlim(f1-25, f1+25)
plt.ylim(-60, 0)
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)
plt.title('Rectangular Window - All Three Frequencies')

plt.subplot(3, 1, 2)
plt.plot(f, fft2db(X1whann), 'C0-', label=f'f1={f1}Hz')
plt.plot(f, fft2db(X2whann), 'C1-', label=f'f2={f2}Hz')
plt.plot(f, fft2db(X3whann), 'C2-', label=f'f3={f3}Hz')
plt.xlim(f1-25, f1+25)
plt.ylim(-60, 0)
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)
plt.title('Hann Window - All Three Frequencies')

plt.subplot(3, 1, 3)
plt.plot(f, fft2db(X1wflattop), 'C0-', label=f'f1={f1}Hz')
plt.plot(f, fft2db(X2wflattop), 'C1-', label=f'f2={f2}Hz')
plt.plot(f, fft2db(X3wflattop), 'C2-', label=f'f3={f3}Hz')
plt.xlim(f1-25, f1+25)
plt.ylim(-60, 0)
plt.legend()
plt.xlabel('f / Hz')
plt.ylabel('A / dB')
plt.grid(True)
plt.title('Flattop Window - All Three Frequencies')

plt.tight_layout()
plt.savefig("all_frequencies.png")

# Calculate and display frequency resolution information
T = N/fs 
delta_f = 1/T 

print(f"Window duration (T): {T:.3f} s")
print(f"Frequency resolution (Δf = 1/T): {delta_f:.3f} Hz")
print(f"Frequency difference between f1 and f2: {abs(f1-f2):.3f} Hz")
print(f"Frequency difference between f1 and f3: {abs(f1-f3):.3f} Hz")