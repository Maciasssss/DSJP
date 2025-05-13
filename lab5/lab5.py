import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram

# --- Variant 8: Sinusoidal Signal in Noise and Welch's Method ---


def random_signal_analysis_variant8():
    """
    Simulates a sinusoidal signal embedded in noise, estimates its statistical properties,
    and uses Welch's method to detect the sinusoidal frequency.
    """
    print("--- Variant 8: Sinusoidal Signal in Noise & Welch's Method ---")

    # Signal Parameters
    fs = 1000  # Sampling frequency (Hz)
    T = 2      # Total time duration (seconds)
    N = int(fs * T)  # Number of samples
    t = np.linspace(0, T, N, endpoint=False)  # Time vector

    # Sinusoidal signal parameters
    f_sin = 50  # Frequency of the sinusoid (Hz)
    A_sin = 1.0  # Amplitude of the sinusoid

    # Noise parameters
    noise_power_factor = 0.5  # Factor to control noise power relative to sinusoid
    # For WGN, variance is related to power. Let's set noise std dev.
    # Power of sinusoid = A_sin^2 / 2 = 1^2 / 2 = 0.5
    # Target noise power = noise_power_factor * (A_sin**2 / 2)
    # For WGN, power = variance = sigma^2
    noise_std_dev = np.sqrt(noise_power_factor * (A_sin**2 / 2))
    # noise_std_dev = 0.5 # Or a fixed value

    # Generate signals
    sinusoid = A_sin * np.sin(2 * np.pi * f_sin * t)
    noise = np.random.normal(loc=0.0, scale=noise_std_dev, size=N)
    noisy_signal = sinusoid + noise

    print(f"Sampling Frequency (fs): {fs} Hz")
    print(f"Signal Duration: {T} s")
    print(f"Number of Samples (N): {N}")
    print(f"Sinusoid Frequency (f_sin): {f_sin} Hz")
    print(f"Sinusoid Amplitude (A_sin): {A_sin}")
    print(f"Noise Standard Deviation: {noise_std_dev:.4f}")
    print(f"Theoretical Sinusoid Power: {A_sin**2 / 2:.4f}")
    print(f"Theoretical Noise Power (Variance): {noise_std_dev**2:.4f}")

    # 1. Estimate Statistical Properties of the noisy signal
    mean_noisy = np.mean(noisy_signal)
    variance_noisy = np.var(noisy_signal)
    std_dev_noisy = np.std(noisy_signal)

    print("\nStatistical Properties of the Noisy Signal:")
    print(f"  Mean: {mean_noisy:.4f}")
    print(f"  Variance: {variance_noisy:.4f}")
    print(f"  Standard Deviation: {std_dev_noisy:.4f}")

    # Theoretical variance of noisy signal (if sinusoid and noise are uncorrelated):
    # Var(sin + noise) = Var(sin) + Var(noise)
    # Var(sin) for a full period or many periods is approx A_sin^2 / 2 (for zero mean)
    # However, for a finite realization, it's better to calculate np.var(sinusoid)
    theoretical_variance_noisy = np.var(sinusoid) + noise_std_dev**2
    print(
        f"  Theoretical Variance (calculated from samples): {theoretical_variance_noisy:.4f}")

    # Plot the signals
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t[:200], sinusoid[:200],
             label=f'Sinusoid ({f_sin} Hz)', color='blue')
    plt.plot(t[:200], noisy_signal[:200],
             label='Noisy Signal', color='red', alpha=0.7)
    plt.title('Sinusoidal Signal and Noisy Signal (Time Domain - First 200 Samples)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # 2. Use Welch's method to detect the sinusoidal frequency
    # Parameters for Welch's method
    nperseg_welch = 256  # Length of each segment
    noverlap_welch = nperseg_welch // 2  # Overlap between segments (e.g., 50%)
    window_welch = 'hann'  # Window function

    frequencies_welch, psd_welch = welch(
        noisy_signal,
        fs=fs,
        window=window_welch,
        nperseg=nperseg_welch,
        noverlap=noverlap_welch,
        # 'density' for PSD (V**2/Hz), 'spectrum' for Power Spectrum (V**2)
        scaling='density'
    )

    plt.subplot(2, 1, 2)
    # plt.semilogy(frequencies_periodogram, psd_periodogram, label='Periodogram PSD', alpha=0.6, color='orange')
    plt.semilogy(frequencies_welch, psd_welch,
                 label=f"Welch's Method PSD (nperseg={nperseg_welch})", color='green')
    plt.title("Power Spectral Density (PSD) of Noisy Signal using Welch's Method")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V^2/Hz)')
    plt.axvline(f_sin, color='purple', linestyle='--',
                label=f'Actual Sinusoid Freq ({f_sin} Hz)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('variant_8_sinusoid_noise_welch.png')
    plt.show()

    # Detect peak frequency from Welch's PSD
    peak_index_welch = np.argmax(psd_welch)
    detected_f_welch = frequencies_welch[peak_index_welch]
    print("\nPSD Estimation using Welch's Method:")
    print(f"  Segment length (nperseg): {nperseg_welch}")
    print(f"  Overlap: {noverlap_welch}")
    print(f"  Window: {window_welch}")
    print(
        f"  Detected peak frequency: {detected_f_welch:.2f} Hz (Actual: {f_sin} Hz)")
    print("-" * 60)


if __name__ == "__main__":
    random_signal_analysis_variant8()
