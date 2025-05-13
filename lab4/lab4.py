import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# --- Part 1: Sampling and Reconstruction - Variant 8 ---


def demonstrate_aliasing_variant8():
    """
    Demonstrates aliasing for a sine wave with f = 50 Hz, sampled at fs = 45 Hz.
    """
    print("--- Part 1: Demonstrating Aliasing (Variant 8) ---")
    f_signal = 50  # Hz
    fs_aliasing = 45  # Hz

    # Time vector for the original signal (plot smoothly)
    # Let's observe for 0.2 seconds to see a few cycles of original and aliased signal
    # 50 Hz signal: period = 1/50 = 0.02s. In 0.2s, we see 10 cycles.
    # Expected aliased frequency: |50 - 1*45| = 5 Hz. Period = 1/5 = 0.2s. In 0.2s, we see 1 cycle.
    t_duration = 0.2
    # High resolution for smooth plot
    t_original = np.linspace(0, t_duration, 500, endpoint=False)
    original_signal = np.sin(2 * np.pi * f_signal * t_original)

    # Time vector for sampling
    t_sampled = np.arange(0, t_duration, 1 / fs_aliasing)
    sampled_signal = np.sin(2 * np.pi * f_signal * t_sampled)

    print(f"Original signal frequency: {f_signal} Hz")
    print(f"Sampling frequency: {fs_aliasing} Hz")
    nyquist_rate = 2 * f_signal
    print(f"Nyquist rate: {nyquist_rate} Hz")
    if fs_aliasing < nyquist_rate:
        print("Aliasing is expected as sampling frequency is less than Nyquist rate.")
        k = 1
        f_aliased_expected = abs(f_signal - k * fs_aliasing)
        print(
            f"Expected principal aliased frequency: |f - k*fs| = |{f_signal} - {k}*{fs_aliasing}| = {f_aliased_expected} Hz")

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(t_original, original_signal,
             label=f'Original Signal ({f_signal} Hz)', color='blue')
    plt.stem(t_sampled, sampled_signal, linefmt='r-', markerfmt='ro', basefmt=" ",
             label=f'Sampled Signal (fs={fs_aliasing} Hz)')
    aliased_signal_shape = np.sin(2 * np.pi * f_aliased_expected * t_original)
    plt.plot(t_original, aliased_signal_shape,
             label=f'Expected Aliased Shape ({f_aliased_expected} Hz)', color='green', linestyle='--')

    plt.title('Variant 8: Demonstration of Aliasing')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig('variant_8_aliasing_demo.png')
    plt.show()
    print("-" * 50)

# --- Part 2: Coding and Decoding - Variant 8 ---


def solve_dct_problem_variant8():
    """
    Applies DCT to the signal [5, 10, 15, 20, 25, 30]
    and reconstructs it with a threshold of 8.
    """
    print("--- Part 2: DCT Compression and Reconstruction (Variant 8) ---")
    original_signal = np.array([5, 10, 15, 20, 25, 30])
    threshold = 8

    print(f"Original Signal: {original_signal}")
    print(f"DCT Threshold: {threshold}")

    # Apply Discrete Cosine Transform (DCT)
    # norm='ortho' makes the DCT orthonormal
    dct_coeffs = dct(original_signal, norm='ortho')
    print(f"DCT Coefficients: {dct_coeffs}")

    # Apply thresholding
    dct_coeffs_thresholded = dct_coeffs.copy()
    dct_coeffs_thresholded[np.abs(dct_coeffs_thresholded) < threshold] = 0
    print(f"Thresholded DCT Coefficients: {dct_coeffs_thresholded}")

    # Reconstruct signal using inverse DCT (IDCT)
    reconstructed_signal = idct(dct_coeffs_thresholded, norm='ortho')
    # Round to a few decimal places for cleaner display, if desired
    reconstructed_signal_rounded = np.round(reconstructed_signal, 2)

    print(f"Reconstructed Signal: {reconstructed_signal}")
    print(f"Reconstructed Signal (Rounded): {reconstructed_signal_rounded}")

    # Calculate Mean Squared Error (MSE) as a measure of distortion
    mse = np.mean((original_signal - reconstructed_signal)**2)
    print(
        f"Mean Squared Error (MSE) between original and reconstructed: {mse:.4f}")
    print("-" * 50)

    # Optional: Plotting for visualization
    plt.figure(figsize=(10, 6))
    n_points = len(original_signal)
    x_axis = np.arange(n_points)

    plt.stem(x_axis - 0.2, original_signal, linefmt='b-',
             markerfmt='bo', basefmt=" ", label='Original Signal')
    plt.stem(x_axis + 0.2, reconstructed_signal_rounded, linefmt='g-',
             markerfmt='gs', basefmt=" ", label=f'Reconstructed (Thresh={threshold})')

    plt.title('Variant 8: DCT Compression and Reconstruction')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.xticks(x_axis)
    plt.legend()
    plt.grid(True)
    plt.savefig('variant_8_dct_reconstruction.png')
    plt.show()


if __name__ == "__main__":
    demonstrate_aliasing_variant8()
    solve_dct_problem_variant8()
