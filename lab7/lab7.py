import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
from scipy.io.wavfile import write as write_wav
import librosa
import librosa.display

# --- Variant 8: Fractal Noise Image and Impulse Train Audio ---
# == PART 1: IMAGE PROCESSING ==


def generate_fractal_noise_image(size=256, sigma_val=8, filename='fractal_noise_image.png'):
    """Generates and saves a fractal-like noise image (smoothed random)."""
    print(
        f"Generating fractal noise image (size={size}x{size}, sigma={sigma_val})...")
    # Generate random noise
    noise = np.random.rand(size, size)
    # Smooth with Gaussian filter to create fractal-like appearance
    fractal_image = gaussian_filter(noise, sigma=sigma_val)
    # Normalize to 0-255 and convert to uint8
    fractal_image_uint8 = ((fractal_image - np.min(fractal_image)) /
                           (np.max(fractal_image) - np.min(fractal_image)) * 255).astype(np.uint8)
    cv2.imwrite(filename, fractal_image_uint8)
    print(f"Fractal noise image saved as {filename}")
    return fractal_image_uint8


def analyze_image(image_path='fractal_noise_image.png'):
    """
    Loads a grayscale image, computes its 2D DFT, visualizes magnitude spectrum,
    applies a band-pass filter, and reconstructs the image.
    """
    print(f"\nAnalyzing image: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(
            f"Image not found at {image_path}. Ensure it's generated first.")

    # 1. Compute 2D Fourier Transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # Shift zero-frequency component to center

    # 2. Visualize Magnitude Spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # +1 to avoid log(0)

    # 3. Apply a Band-Pass Filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    r_inner = 20  # Inner radius (removes low frequencies)
    # Outer radius (removes high frequencies, but keeps more than a pure LPF)
    r_outer = 80
    # For fractal noise, high frequencies are characteristic.

    mask_bp = np.zeros((rows, cols), dtype=np.float32)
    # Outer circle (everything inside is 1)
    cv2.circle(mask_bp, (ccol, crow), r_outer, 1, thickness=-1)
    # Inner circle (set center to 0 again)
    cv2.circle(mask_bp, (ccol, crow), r_inner, 0, thickness=-1)
    # This creates a band-pass mask

    fshift_bp_filtered = fshift * mask_bp

    # 4. Reconstruct the Image from the Filtered Spectrum
    f_ishift_bp = np.fft.ifftshift(fshift_bp_filtered)
    img_back_bp = np.fft.ifft2(f_ishift_bp)
    img_back_bp = np.abs(img_back_bp)
    # Normalize for display
    img_back_bp_display = ((img_back_bp - np.min(img_back_bp)) /
                           (np.max(img_back_bp) - np.min(img_back_bp) + 1e-6) * 255).astype(np.uint8)

    # Plotting
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(mask_bp, cmap='gray')
    plt.title(f'Band-Pass Mask (r_in={r_inner}, r_out={r_outer})')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(img_back_bp_display, cmap='gray')
    plt.title('Band-Pass Filtered Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('variant_8_image_analysis.png')
    plt.show()
    print("Image analysis plots saved as variant_8_image_analysis.png")


# == PART 2: AUDIO PROCESSING ==
def generate_impulse_train_audio(fs=44100, duration=5, interval_sec=1, amplitude=0.8, filename='impulse_train_audio.wav'):
    """Generates and saves an impulse train audio signal."""
    print(
        f"\nGenerating impulse train audio (fs={fs}, duration={duration}s, interval={interval_sec}s)...")
    num_samples = int(fs * duration)
    signal = np.zeros(num_samples)

    # interval_samples = int(fs * interval_sec)
    # The template example code places impulses at i * fs, which for interval=1 means 0, 1*fs, 2*fs...
    # Let's stick to the template's logic for `interval` being the loop step.
    # i will be 0, 1, 2, 3, 4 for 5s duration, 1s interval
    for i in range(0, int(duration), interval_sec):
        sample_index = int(i * fs)
        if sample_index < num_samples:
            signal[sample_index] = amplitude

    signal_pcm = (signal * 32767).astype(np.int16)
    write_wav(filename, fs, signal_pcm)
    print(f"Impulse train audio saved as {filename}")
    return signal, fs


def analyze_audio(audio_signal, fs, filename_prefix='variant_8'):
    """
    Analyzes an audio signal: plots time-domain waveform, its FFT magnitude spectrum,
    and its spectrogram.
    """
    print("\nAnalyzing audio signal...")
    duration = len(audio_signal) / fs
    t = np.linspace(0, duration, len(audio_signal), endpoint=False)

    # 1. Time-Domain Representation
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(audio_signal, sr=fs, ax=plt.gca())
    # plt.plot(t, audio_signal) # Alternative basic plot
    plt.title('Time Domain Signal (Impulse Train)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # 2. Frequency-Domain Representation (Overall FFT)
    Y = np.fft.fft(audio_signal)
    freqs = np.fft.fftfreq(len(Y), 1/fs)
    positive_freq_indices = np.where(freqs >= 0)
    freqs_pos = freqs[positive_freq_indices]
    Y_mag_pos = np.abs(Y[positive_freq_indices])

    plt.subplot(3, 1, 2)
    num_points_to_stem = 2000
    if len(freqs_pos) > num_points_to_stem:
        step = len(freqs_pos) // num_points_to_stem
        plt.stem(freqs_pos[::step], Y_mag_pos[::step],
                 basefmt=" ")
        plt.title(f'Magnitude Spectrum (FFT - showing every {step}th point)')
    else:
        plt.stem(freqs_pos, Y_mag_pos, basefmt=" ")
        plt.title('Magnitude Spectrum (FFT of entire signal)')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    # 3. Time-Frequency Representation (Spectrogram)
    # STFT parameters
    n_fft = 2048  # Window length for FFT
    hop_length = 512  # Number of samples between successive STFT columns

    D_stft = librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D_stft), ref=np.max)

    plt.subplot(3, 1, 3)
    img = librosa.display.specshow(S_db, sr=fs, hop_length=hop_length,
                                   x_axis='time', y_axis='hz', cmap='magma', ax=plt.gca())
    plt.colorbar(img, format='%+2.0f dB', ax=plt.gca())
    plt.title('Spectrogram (STFT)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_audio_analysis.png')
    plt.show()
    print(
        f"Audio analysis plots saved as {filename_prefix}_audio_analysis.png")


if __name__ == "__main__":
    generated_image = generate_fractal_noise_image(size=256, sigma_val=8)
    analyze_image(image_path='fractal_noise_image.png')

    generated_audio, sample_rate = generate_impulse_train_audio(
        fs=44100, duration=3, interval_sec=1)
    analyze_audio(generated_audio, sample_rate)

    print("\nVariant 8 execution complete.")
