import numpy as np
import matplotlib.pyplot as plt

# Given vector x_μ
x_mu = np.array([6, 2, 4, 4, 4, 5, 0, 0, 0, 0])
M = len(x_mu)  # Length of the vector

# Function to compute the W matrix for IDFT
def compute_W_matrix(N):
    W = np.zeros((N, N), dtype=complex)
    for k in range(N):
        for n in range(N):
            W[k, n] = np.exp(2j * np.pi * k * n / N)
    return W / N  # Normalize by 1/N for IDFT

# Function to compute the K matrix for IDFT
def compute_K_matrix(M, N):
    K = np.zeros((N, M), dtype=complex)
    for n in range(N):
        for mu in range(M):
            if mu < N:
                K[n, mu] = 1 if n == mu else 0
            else:
                K[n, mu] = 0
    return K

# Plot the synthesized signal for different values of N
plt.figure(figsize=(15, 10))

N_values = [4, 6, 8, 10, 16, 32]
for i, N in enumerate(N_values):
    # Compute W matrix
    W = compute_W_matrix(N)
    
    # If M > N, we'll truncate x_mu; if M < N, we'll pad x_mu
    x_mu_adjusted = np.zeros(N, dtype=complex)
    x_mu_adjusted[:min(M, N)] = x_mu[:min(M, N)]
    
    # Compute synthesized signal
    x_n = W @ x_mu_adjusted
    
    # Plot the real part of the synthesized signal
    plt.subplot(len(N_values), 2, 2*i+1)
    plt.stem(np.arange(N), np.real(x_n))
    plt.title(f'Real part of synthesized signal (N={N})')
    plt.xlabel('n')
    plt.ylabel('x[n]')
    plt.grid(True)
    
    # Plot the imaginary part of the synthesized signal
    plt.subplot(len(N_values), 2, 2*i+2)
    plt.stem(np.arange(N), np.imag(x_n))
    plt.title(f'Imaginary part of synthesized signal (N={N})')
    plt.xlabel('n')
    plt.ylabel('x[n]')
    plt.grid(True)

plt.tight_layout()

# Display W and K matrices for a specific case (N=10, which matches the length of x_mu)
N_for_display = 10
W_display = compute_W_matrix(N_for_display)
K_display = compute_K_matrix(M, N_for_display)

print(f"W matrix for N={N_for_display}:")
np.set_printoptions(precision=3, suppress=True)
print(W_display)

print(f"\nK matrix for N={N_for_display}, M={M}:")
print(K_display)

# Calculate and display the final synthesized signal for N=10
x_n_display = W_display @ x_mu[:N_for_display]
print("\nSynthesized signal for N=10:")
print("Real part:", np.real(x_n_display))
print("Imaginary part:", np.imag(x_n_display))

# Plot the complete synthesized signal for N=10
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.stem(np.arange(N_for_display), np.real(x_n_display))
plt.title('Real part of synthesized signal (N=10)')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.stem(np.arange(N_for_display), np.imag(x_n_display))
plt.title('Imaginary part of synthesized signal (N=10)')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid(True)

plt.tight_layout()
plt.show()