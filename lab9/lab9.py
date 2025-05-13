import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import butter, lfilter

# --- Variant 8: Real-Time ECG and Kalman-Bucy Filtering ---

# == PART A: Real-Time ECG Signal Processing ==


def synthetic_ecg_part_a(fs, duration, heart_rate=70):
    """Generates a synthetic ECG-like signal for Part A."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    p_wave = 0.1 * np.sin(2 * np.pi * (heart_rate / 60) *
                          # Simplified P
                          0.5 * (t % (60/heart_rate)) - np.pi/2)
    qrs_complex = 1.0 * np.sin(2 * np.pi * (heart_rate / 60) * (t % (60/heart_rate))) * \
        np.exp(-50 * ((t % (60/heart_rate)) - 0.25)**2)  # QRS (dominant)
    t_wave = 0.3 * np.sin(2 * np.pi * (heart_rate / 60) *
                          # Simplified T
                          0.7 * (t % (60/heart_rate)) + np.pi/1.5)

    # Simulate beat-to-beat placement
    ecg_template = qrs_complex  # Start with QRS

    # Simulate QRS complex (approx. 1 Hz for HR=60)
    qrs = 0.6 * np.sin(2 * np.pi * (heart_rate / 60) * t)
    # Simulate T-wave (often around double frequency or related harmonic of QRS)
    # The example uses 2*heart_rate/60 which makes it a harmonic.
    t_wave_like = 0.2 * np.sin(2 * np.pi * 2 * (heart_rate / 60) * t)
    # Add some baseline noise
    noise = 0.1 * np.random.randn(len(t))

    ecg = qrs + t_wave_like + noise
    return t, ecg


def bandpass_filter_ecg(signal_block, fs, lowcut=0.5, highcut=40.0, order=4):
    """Applies a Butterworth bandpass filter to a signal block (causal lfilter)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if low <= 0:
        low = 1e-4
    if high >= 1:
        high = 1 - 1e-4
    if low >= high:
        print(
            f"Warning: lowcut ({lowcut} Hz) is not less than highcut ({highcut} Hz) after Nyquist normalization. Adjusting lowcut.")
        low = high / 2
        if low <= 0:
            low = 1e-4

    b, a = butter(order, [low, high], btype='band')

    filtered_block = lfilter(b, a, signal_block)
    return filtered_block


def simulate_real_time_ecg_processing(fs, duration, block_size):
    """Simulates real-time ECG filtering block by block."""
    print("--- Part A: Simulating Real-Time ECG Filtering ---")
    print(
        f"Params: Fs={fs}Hz, Duration={duration}s, Block Size={block_size} samples")

    t_vec, ecg_signal_raw = synthetic_ecg_part_a(
        fs, duration, heart_rate=70)  # Simulate with ~70 BPM

    total_samples = len(ecg_signal_raw)
    num_blocks = total_samples // block_size

    processed_signal_list = []
    t_axis_list = []

    # Define filter parameters (common for ECG)
    lowcut_filt = 0.5  # Hz
    highcut_filt = 40.0  # Hz
    filter_order = 4

    print(f"Filtering with bandpass: {lowcut_filt}-{highcut_filt} Hz")

    plt.figure(figsize=(12, 8))
    plt.ion()

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size

        current_block_raw = ecg_signal_raw[start_idx:end_idx]
        current_block_time = t_vec[start_idx:end_idx]

        # Apply bandpass filter to the current block
        current_block_filtered = bandpass_filter_ecg(current_block_raw, fs,
                                                     lowcut=lowcut_filt, highcut=highcut_filt,
                                                     order=filter_order)

        processed_signal_list.extend(current_block_filtered)
        t_axis_list.extend(current_block_time)

        # --- Real-time-like plotting (can be slow) ---
        plt.clf()

        # Plot processed signal so far
        if t_axis_list:
            plt.plot(t_axis_list, processed_signal_list,
                     label='Filtered ECG (Real-Time Sim)', color='blue')

        plt.title(
            f"Real-Time ECG Filtering Simulation (Block {i+1}/{num_blocks})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.xlim([0, duration])  # Keep x-axis fixed
        # Dynamic y-axis or fixed based on expected signal range
        if processed_signal_list:
            min_val = np.min(processed_signal_list)
            max_val = np.max(processed_signal_list)
            padding = (max_val - min_val) * 0.1
            if max_val - min_val > 1e-6:
                plt.ylim([min_val - padding, max_val + padding])
            else:
                plt.ylim([-1, 1])

        plt.legend(loc='upper right')
        plt.grid(True)
        plt.pause(0.01)
        # --- End of real-time-like plotting ---

    plt.ioff()  # Turn off interactive mode
    plt.figure(figsize=(12, 6))
    plt.plot(t_vec, ecg_signal_raw, label='Original Raw ECG',
             color='lightcoral', alpha=0.6)
    plt.plot(t_axis_list, processed_signal_list,
             label='Completed Filtered ECG', color='blue')
    plt.title('Final: Original Raw vs. Completed "Real-Time" Filtered ECG')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig('variant_8_ecg_realtime_final.png')
    plt.show()
    print("Part A: Real-time ECG simulation complete. Final plot saved.")


# == PART B: Kalman-Bucy Filtering ==

def simulate_kalman_bucy_scalar(A, C, Q_noise, R_noise, P0, x0_true, x0_hat, T_duration, dt):
    """
    Simulates a scalar continuous-time system with Kalman-Bucy filter.
    System: dx/dt = A*x + w  (w ~ N(0,Q))
    Measurement: y = C*x + v (v ~ N(0,R))
    """
    print("\n--- Part B: Simulating Kalman-Bucy Filter ---")
    print(
        f"Params: A={A}, C={C}, Q={Q_noise}, R={R_noise}, P0={P0}, T_dur={T_duration}s, dt={dt}s")

    N_steps = int(T_duration / dt)

    x_true_history = np.zeros(N_steps)
    x_hat_history = np.zeros(N_steps)
    P_history = np.zeros(N_steps)
    K_history = np.zeros(N_steps)

    x_true = x0_true
    x_hat = x0_hat
    P = P0

    time_vec = np.linspace(0, T_duration, N_steps, endpoint=False)

    for k in range(N_steps):
        # 1. Simulate True System (Euler-Maruyama for stochastic differential equation)
        # Process noise sample, scaled by sqrt(dt)
        w_k = np.random.normal(0, np.sqrt(Q_noise * dt))
        # Euler step for dx = A*x*dt + dw
        x_true = x_true + (A * x_true) * dt + w_k

        noise_process_increment = np.random.normal(0, np.sqrt(Q_noise * dt))
        x_true = x_true + (A * x_true) * dt + noise_process_increment

        # 2. Simulate Measurement
        # Measurement noise. If R is variance of v_k, then no /dt.
        v_k = np.random.normal(0, np.sqrt(R_noise / dt))
        v_k_measurement = np.random.normal(0, np.sqrt(R_noise))
        y_k = C * x_true + v_k_measurement

        # 3. Kalman-Bucy Filter Update (Continuous equations discretized via Euler)
        # Kalman Gain (continuous form)
        K = P * C * (1/R_noise)  # Since R is scalar, R_inv = 1/R

        # State Estimate Update (dx_hat/dt = A*x_hat + K*(y - C*x_hat))
        dx_hat_dt = A * x_hat + K * (y_k - C * x_hat)
        x_hat = x_hat + dx_hat_dt * dt

        # Covariance Update (dP/dt = A*P + P*A^T + Q - K*C*P)
        # For scalar: dP/dt = 2*A*P + Q - K*C*P = 2*A*P + Q - (P*C/R)*C*P
        dP_dt = 2 * A * P + Q_noise - K * C * P
        P = P + dP_dt * dt
        if P < 0:
            P = 1e-9  # Covariance must be non-negative

        # Store history
        x_true_history[k] = x_true
        x_hat_history[k] = x_hat
        P_history[k] = P
        K_history[k] = K

    # Plotting Kalman-Bucy results
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(time_vec, x_true_history,
             label='True State (x_true)', color='blue')
    plt.plot(time_vec, x_hat_history, label='Estimated State (x_hat)',
             color='red', linestyle='--')
    plt.title('Kalman-Bucy Filter: State Estimation')
    plt.xlabel('Time (s)')
    plt.ylabel('State Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_vec, P_history, label='Error Covariance (P)', color='green')
    plt.title('Error Covariance P(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Covariance P')
    if np.all(P_history > 0):
        plt.axhline(P_history[-1], color='gray', linestyle=':',
                    label=f'Steady State P ≈ {P_history[-1]:.3f}')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_vec, K_history, label='Kalman Gain (K)', color='purple')
    plt.title('Kalman Gain K(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Gain K')
    if np.all(K_history > -np.inf):
        plt.axhline(K_history[-1], color='gray', linestyle=':',
                    label=f'Steady State K ≈ {K_history[-1]:.3f}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('variant_8_kalman_bucy_filter.png')
    plt.show()
    print("Part B: Kalman-Bucy filter simulation complete. Plot saved.")


if __name__ == "__main__":
    # Part A Parameters for Variant 8
    ecg_fs_v8 = 300       # Hz
    ecg_duration_v8 = 14  # s
    ecg_block_size_v8 = 150  # samples
    simulate_real_time_ecg_processing(
        fs=ecg_fs_v8, duration=ecg_duration_v8, block_size=ecg_block_size_v8)

    # Part B Parameters for Variant 8
    # Kalman-Bucy: ẋ = -x + w, y = x + v, Q = 0.6, R = 1
    kb_A = -1.0
    kb_C = 1.0
    kb_Q = 0.6
    kb_R = 1.0
    kb_P0 = 1.0       # Initial error covariance guess
    kb_x0_true = 0.0  # Initial true state
    kb_x0_hat = 0.0   # Initial state estimate
    kb_T_sim = 10.0   # Simulation duration for Kalman-Bucy (e.g., 10s)
    kb_dt = 0.01      # Time step for numerical integration

    simulate_kalman_bucy_scalar(A=kb_A, C=kb_C, Q_noise=kb_Q, R_noise=kb_R,
                                P0=kb_P0, x0_true=kb_x0_true, x0_hat=kb_x0_hat,
                                T_duration=kb_T_sim, dt=kb_dt)

    print("\nVariant 8 Real-Time Processing and Kalman-Bucy Lab complete.")
