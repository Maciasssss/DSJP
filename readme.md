# Digital Signal Processing Laboratory Exercises

![DSP Banner](https://via.placeholder.com/800x200.png?text=Digital+Signal+Processing+Lab)

Welcome to the repository for my Digital Signal Processing (DSP) laboratory exercises completed as part of the **["Analog and Digital Electronic Circuits"]** course at the **[University of Bielsko-Biala]**.

This repository contains Python implementations and corresponding reports for a series of laboratory assignments covering fundamental and advanced DSP concepts.

## 📚 Covered Laboratory Topics

This collection of labs explores a diverse range of topics within Digital Signal Processing. Each lab builds upon previous concepts, providing a comprehensive understanding of how signals are analyzed, processed, and modeled.

---

### Examples 7 & 8: Sampling, Reconstruction, Coding & Decoding

- **Lab 7: Sampling and Reconstruction of Signals**
  - Nyquist-Shannon Sampling Theorem
  - Aliasing effects and demonstration
  - Signal reconstruction techniques (e.g., sinc interpolation)
  - Generation and analysis of various waveforms (sine, cosine, square, triangular, sawtooth)
- **Lab 8: Coding and Decoding Digital Signals**
  - Principles of signal coding and decoding
  - Compression algorithms: Delta Encoding, Quantization, Transform-Based Compression (DCT)
  - Analysis of compression metrics (SNR, Compression Ratio)
  - Trade-off analysis between distortion and compression.

## 🛠️ Tools and Libraries Used

- **Python 3.10 to 13**
- **NumPy:** For numerical computations and array manipulation.
- **SciPy:** For scientific and technical computing, including:
  - `scipy.signal`: For filters (Butterworth, `lfilter`, `welch`, `chirp`, `square`, `sawtooth`), STFT.
  - `scipy.fftpack` / `scipy.fft`: For FFT, DCT, IDCT.
  - `scipy.ndimage`: For image filtering (e.g., `gaussian_filter`).
  - `scipy.io.wavfile`: For reading/writing WAV audio files.
- **Matplotlib:** For generating static, animated, and interactive visualizations.
- **OpenCV (`cv2`):** For image loading, saving, and basic image processing operations.
- **Librosa:** For audio analysis, STFT, spectrogram display, and waveform visualization.
- **Statsmodels:** For statistical modeling, including ARMA/ARIMA model fitting and diagnostics.
- **NeuroKit2:** For biomedical signal processing, particularly ECG simulation and R-peak detection.
- **Spectrum:** (Potentially used for some AR PSD methods like Burg - check specific lab if used)

## 🚀 Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-dsp-repo-name.git
    cd your-dsp-repo-name
    ```
2.  **Navigate to specific lab directories** (if applicable) and run the Python scripts. Each lab typically includes a main script (e.g., `labX.py`) and may generate plots and console output.

## 👤 Author

- **[Maciej Kos]**
- **[maciek_k112@wp.pl]**

## 🙏 Acknowledgements

- Prof. dr hab. Vasyl Martsenyuk for the laboratory instructions and guidance.
- Department of Computer Science and Automatics, University of Bielsko-Biala.

---
