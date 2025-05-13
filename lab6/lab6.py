import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf

# --- Variant 8: MA(3) Data Generation, Model Fitting, and Residual Analysis ---


def signal_modeling_variant8():
    """
    Generates MA(3) data, fits MA(2), MA(3), and ARMA(1,1) models,
    and analyzes residual autocorrelations.
    """
    print("--- Variant 8: MA(3) Data, Model Fitting, and Residual Analysis ---")

    # 1. Generate MA(3) Data
    np.random.seed(42)
    N = 1000

    # MA(3) parameters: x[n] = w[n] + b1*w[n-1] + b2*w[n-2] + b3*w[n-3]
    # ArmaProcess uses [1, b1, b2, b3] for MA coefficients (b0=1 is implicit for the current noise term)
    ma_coeffs_true = np.array([0.6, 0.4, 0.2])  # b1, b2, b3
    ar_coeffs_true = np.array([1])
    # For ArmaProcess, MA part needs a leading 1 for b0
    ma_process_coeffs = np.concatenate(([1], ma_coeffs_true))

    ma3_process = ArmaProcess(ar_coeffs_true, ma_process_coeffs)
    # Generate data - note: ArmaProcess expects AR coefficients first, then MA.
    # For pure MA(q), AR part is empty, MA part is [1, b1, ..., bq].
    # For pure AR(p), MA part is [1], AR part is [1, -a1, ..., -ap].
    # The lfilter uses [1] for the 'a' (denominator) if it's a pure MA process.
    # The 'b' (numerator) will be [1, b1, b2, b3].

    w = np.random.normal(loc=0, scale=1, size=N)
    # Simulating MA(3) using lfilter: x[n] = w[n] + 0.6w[n-1] + 0.4w[n-2] + 0.2w[n-3]
    # b coefficients for lfilter are [b0, b1, b2, b3] where b0=1
    b_lfilter = [1, 0.6, 0.4, 0.2]
    ma3_data = lfilter(b_lfilter, [1], w)

    print(f"Generated MA(3) data with N={N} samples.")
    print(f"True MA coefficients (b1, b2, b3): {ma_coeffs_true}")

    plt.figure(figsize=(12, 4))
    plt.plot(ma3_data)
    plt.title('Generated MA(3) Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig('variant_8_ma3_data.png')
    plt.show()

    # 2. Fit Models
    #   a) Fit MA(2) model
    print("\nFitting MA(2) model...")
    # For ARIMA(p,d,q), MA(q) is order=(0,0,q)
    model_ma2 = ARIMA(ma3_data, order=(0, 0, 2))
    results_ma2 = model_ma2.fit()
    print(results_ma2.summary())
    residuals_ma2 = results_ma2.resid

    #   b) Fit MA(3) model
    print("\nFitting MA(3) model...")
    model_ma3 = ARIMA(ma3_data, order=(0, 0, 3))
    results_ma3 = model_ma3.fit()
    print(results_ma3.summary())
    residuals_ma3 = results_ma3.resid

    #   c) Fit ARMA(1,1) model
    print("\nFitting ARMA(1,1) model...")
    # ARMA(p,q) is order=(p,0,q)
    model_arma11 = ARIMA(ma3_data, order=(1, 0, 1))
    results_arma11 = model_arma11.fit()
    print(results_arma11.summary())
    residuals_arma11 = results_arma11.resid

    # 3. Analyze Residual Autocorrelations
    lags_acf = 30

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plot_acf(residuals_ma2, lags=lags_acf, ax=plt.gca(),
             title='ACF of Residuals - MA(2) Model')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plot_acf(residuals_ma3, lags=lags_acf, ax=plt.gca(),
             title='ACF of Residuals - MA(3) Model')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plot_acf(residuals_arma11, lags=lags_acf, ax=plt.gca(),
             title='ACF of Residuals - ARMA(1,1) Model')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('variant_8_residual_acf.png')
    plt.show()

    print("\n--- AIC Values ---")
    print(f"MA(2) Model AIC: {results_ma2.aic:.2f}")
    print(f"MA(3) Model AIC: {results_ma3.aic:.2f}")
    print(f"ARMA(1,1) Model AIC: {results_arma11.aic:.2f}")

    print("\n--- Ljung-Box Test for Residuals (example for MA(3) model) ---")
    lb_test_ma3 = sm.stats.acorr_ljungbox(
        residuals_ma3, lags=[10], return_df=True)  # Test at lag 10
    print(f"Ljung-Box test for MA(3) residuals (lags=10):")
    print(f"  LB-statistic: {lb_test_ma3['lb_stat'].iloc[0]:.2f}")
    print(f"  p-value: {lb_test_ma3['lb_pvalue'].iloc[0]:.2f}")

    if lb_test_ma3['lb_pvalue'].iloc[0] > 0.05:
        print("  Ljung-Box test suggests residuals for MA(3) are likely white noise (good fit).")
    else:
        print("  Ljung-Box test suggests significant autocorrelation remains in MA(3) residuals (potential model inadequacy).")

    print("-" * 60)


if __name__ == "__main__":
    signal_modeling_variant8()
