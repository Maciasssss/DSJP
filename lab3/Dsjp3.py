import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# Generate a noisy sinusoidal signal
fs = 1000  # Sampling frequency
T = 1  # Duration in seconds
t = np.linspace(0, T, fs, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sinusoid
noise = 0.5 * np.random.randn(len(t))  # Gaussian noise
noisy_signal = signal + noise

# FIR Filter Implementation
b_fir = [1, 0, 1]  # FIR filter coefficients
def fir_filter(x, b):
    return lfilter(b, [1], x)
filtered_fir = fir_filter(noisy_signal, b_fir)

# IIR Filter Implementation
b_iir = [0.5, 0.5]  # IIR numerator coefficients
a_iir = [1, -0.3]  # IIR denominator coefficients
def iir_filter(x, b, a):
    return lfilter(b, a, x)
filtered_iir = iir_filter(noisy_signal, b_iir, a_iir)

# Adaptive LMS Filter Implementation
mu = 0.05  # Step size
M = 5  # Filter length
def lms_filter(x, d, M, mu):
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(M, N):
        x_n = x[n-M:n][::-1]
        y[n] = np.dot(w, x_n)
        e[n] = d[n] - y[n]
        w += 2 * mu * e[n] * x_n
    return y
filtered_lms = lms_filter(noisy_signal, signal, M, mu)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(t, filtered_fir, label='FIR Filtered', color='r')
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(t, filtered_iir, label='IIR Filtered', color='g')
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(t, filtered_lms, label='LMS Filtered', color='m')
plt.legend()
plt.xlabel("Time [s]")
plt.show()
