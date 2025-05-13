import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import neurokit2 as nk

# --- Variant 8: ECG Signal Processing ---


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Applies a Butterworth bandpass filter to the signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Ensure low and high are valid and low < high
    if low <= 0:
        low = 1e-4  # Avoid zero or negative lowcut
    if high >= 1:
        high = 1 - 1e-4  # Avoid highcut >= Nyquist
    if low >= high:
        print(
            f"Warning: lowcut ({lowcut} Hz) is not less than highcut ({highcut} Hz) after Nyquist normalization. Adjusting lowcut.")
        low = high / 2
        if low <= 0:
            low = 1e-4

    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y


def calculate_instantaneous_hr(r_peaks_indices, fs):
    """
    Calculates instantaneous heart rate from R-R intervals.
    Returns two arrays: time points for HR and HR values.
    """
    if len(r_peaks_indices) < 2:
        return np.array([]), np.array([])

    # Differences between consecutive R-peaks in samples
    rr_intervals_samples = np.diff(r_peaks_indices)
    rr_intervals_sec = rr_intervals_samples / fs

    instantaneous_hr_bpm = 60.0 / rr_intervals_sec  # HR in beats per minute

    # Time points for HR: typically at the location of the second R-peak of each interval
    hr_time_points_samples = r_peaks_indices[1:]
    hr_time_points_sec = hr_time_points_samples / fs

    return hr_time_points_sec, instantaneous_hr_bpm


def ecg_processing_variant8():
    """
    Performs ECG signal processing for Variant 8.
    - Simulates ECG
    - Applies bandpass filter
    - Detects R-peaks
    - Calculates and plots heart rate
    """
    print("--- Variant 8: ECG Signal Processing ---")

    # Variant 8 Parameters
    duration = 14  # seconds
    fs = 250       # Hz (Sampling Rate)
    lowcut = 0.8   # Hz (Filter low cutoff)
    highcut = 42   # Hz (Filter high cutoff)
    filter_order = 4  # Butterworth filter order

    print(
        f"Parameters: Duration={duration}s, Sampling Rate={fs}Hz, Filter={lowcut}-{highcut}Hz")

    # 1. Simulate ECG signal
    # NeuroKit2 can simulate ECG with some variability
    # We can also set a heart_rate for simulation if desired, e.g., heart_rate=70
    # For more natural variability, nk.ecg_simulate is good.
    ecg_signal_raw = nk.ecg_simulate(
        duration=duration, sampling_rate=fs, heart_rate=75, noise=0.05)
    time_vector = np.linspace(0, duration, len(ecg_signal_raw), endpoint=False)

    print(f"Generated raw ECG signal with {len(ecg_signal_raw)} samples.")

    # 2. Apply bandpass filter
    ecg_signal_filtered = bandpass_filter(
        ecg_signal_raw, lowcut, highcut, fs, order=filter_order)
    print(f"Applied bandpass filter ({lowcut}-{highcut} Hz).")

    # 3. Detect R-peaks (using NeuroKit2 on the filtered signal)
    _, rpeaks_info_filtered = nk.ecg_peaks(
        ecg_signal_filtered, sampling_rate=fs, method="neurokit")
    r_peaks_indices_filtered = rpeaks_info_filtered['ECG_R_Peaks']

    # For comparison, detect R-peaks on raw signal as well
    _, rpeaks_info_raw = nk.ecg_peaks(
        ecg_signal_raw, sampling_rate=fs, method="neurokit")
    r_peaks_indices_raw = rpeaks_info_raw['ECG_R_Peaks']

    print(f"Detected {len(r_peaks_indices_raw)} R-peaks in raw ECG.")
    print(f"Detected {len(r_peaks_indices_filtered)} R-peaks in filtered ECG.")

    # 4. Calculate and plot heart rate over time (from filtered signal's R-peaks)
    hr_time_points, instantaneous_hr = calculate_instantaneous_hr(
        r_peaks_indices_filtered, fs)

    # 5. Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot Raw ECG and its R-peaks
    axs[0].plot(time_vector, ecg_signal_raw,
                label='Raw ECG Signal', color='lightblue', alpha=0.8)
    if len(r_peaks_indices_raw) > 0:
        axs[0].plot(time_vector[r_peaks_indices_raw], ecg_signal_raw[r_peaks_indices_raw],
                    'x', color='red', markersize=8, label='R-peaks (Raw)')
    axs[0].set_title(f'Raw ECG Signal (Duration: {duration}s, FS: {fs}Hz)')
    axs[0].set_ylabel('Amplitude (mV or unitless)')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Filtered ECG and its R-peaks
    axs[1].plot(time_vector, ecg_signal_filtered,
                label=f'Filtered ECG ({lowcut}-{highcut} Hz)', color='green')
    if len(r_peaks_indices_filtered) > 0:
        axs[1].plot(time_vector[r_peaks_indices_filtered], ecg_signal_filtered[r_peaks_indices_filtered],
                    'o', color='red', markersize=6, label='R-peaks (Filtered)')
    axs[1].set_title(f'Bandpass Filtered ECG Signal ({lowcut}-{highcut} Hz)')
    axs[1].set_ylabel('Amplitude (mV or unitless)')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Instantaneous Heart Rate
    if len(hr_time_points) > 0:
        axs[2].plot(hr_time_points, instantaneous_hr, label='Instantaneous Heart Rate',
                    marker='.', linestyle='-', color='purple')
        mean_hr = np.mean(instantaneous_hr)
        axs[2].axhline(mean_hr, color='gray', linestyle='--',
                       label=f'Mean HR: {mean_hr:.2f} BPM')
        axs[2].set_title('Instantaneous Heart Rate (from filtered ECG)')
        axs[2].set_ylabel('Heart Rate (BPM)')
        axs[2].legend()
    else:
        axs[2].set_title(
            'Instantaneous Heart Rate (Not enough R-peaks detected)')
        axs[2].text(0.5, 0.5, 'No HR data to plot', horizontalalignment='center',
                    verticalalignment='center', transform=axs[2].transAxes)

    axs[2].set_xlabel('Time (s)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig('variant_8_ecg_analysis.png')
    plt.show()

    print("\n--- Comments on Filtering Effects ---")
    if len(r_peaks_indices_raw) != len(r_peaks_indices_filtered):
        print(
            f"Filtering changed the number of detected R-peaks (Raw: {len(r_peaks_indices_raw)}, Filtered: {len(r_peaks_indices_filtered)}).")
    else:
        print("Filtering did not change the number of detected R-peaks.")

    if len(r_peaks_indices_filtered) > 0:
        mean_r_amp_raw = np.mean(ecg_signal_raw[r_peaks_indices_raw]) if len(
            r_peaks_indices_raw) > 0 else float('nan')
        mean_r_amp_filtered = np.mean(
            ecg_signal_filtered[r_peaks_indices_filtered])
        print(f"Mean R-peak amplitude in raw signal: {mean_r_amp_raw:.4f}")
        print(
            f"Mean R-peak amplitude in filtered signal: {mean_r_amp_filtered:.4f}")
        if not np.isnan(mean_r_amp_raw) and mean_r_amp_filtered > mean_r_amp_raw * 0.8:
            print(
                "Filtering maintained or enhanced R-peak prominence relative to baseline noise.")
        else:
            print(
                "Filtering might have attenuated R-peaks or R-peak detection could be less robust.")
    else:
        print("No R-peaks detected in filtered signal to comment on amplitude.")

    print("Bandpass filtering is crucial for ECG processing because:")
    print("1. Removes baseline wander (low-frequency noise, e.g., due to respiration or body movement). The lowcut frequency (0.8 Hz) addresses this.")
    print("2. Removes high-frequency noise (e.g., muscle artifacts (EMG), power line interference if not 50/60Hz, or other instrumentation noise). The highcut frequency (42 Hz) addresses this.")
    print("By removing these noise components, the QRS complex, particularly the R-peak, becomes more prominent and easier to detect accurately, leading to more reliable heart rate estimation and morphological analysis.")
    print("However, inappropriate filter cutoffs or order can distort the ECG waveform, potentially affecting diagnostic features or R-peak detection. The chosen range 0.8-42 Hz is a common choice for general ECG analysis.")
    print("-" * 60)


if __name__ == "__main__":
    ecg_processing_variant8()
    print("\nVariant 8 ECG processing complete.")
