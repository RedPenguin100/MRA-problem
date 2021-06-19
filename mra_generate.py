import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

np.random.seed(42)


def generate_data(x: np.array, N, sigma):
    """
    :param x: signal
    :param N: measurements amount
    :param sigma: standard deviation
    :return:
    """
    global INDEXES
    L = x.size
    group_members = np.random.randint(1, L, N)

    noise = np.random.normal(scale=sigma, size=(N, L))
    data = np.zeros((N, L))
    binomial = np.random.binomial(1, 0.5, N)
    one = binomial == 1
    zero = binomial == 0
    data[one] = x.take(INDEXES[group_members[one]]) + noise[one]
    data[zero] = x + noise[zero]

    return data


def signal_to_noise_ratio(x, sigma):
    return np.power(np.linalg.norm(x) / sigma)


def signal_mean_from_data(data):
    """
    :note: The mean of each row separately, and then
    averaging the results, is the same result as averaging
    everything together.
    """
    return np.mean(data)


def signal_first_moment(data):
    return np.mean(data, axis=0)


def signal_power_spectrum_from_data(data, sigma):
    N, L = data.shape
    power_spectra = np.power(np.abs(np.fft.fft(data)), 2.0)
    return np.mean(power_spectra, axis=0) - np.power(sigma, 2.0) * L


def signal_mu_2_from_data(data, sigma):
    N, L = data.shape
    outer_products = np.einsum('bi,bo->bio', data, data)
    return np.mean(outer_products, axis=0) - np.power(sigma, 2.0) * np.eye(L)


def signal_observation_error(signal, observation):
    L = signal.shape[0]
    assert L == observation.shape[0]
    error = np.inf
    for i in range(L):
        new_error = np.linalg.norm(signal - observation.take(INDEXES[i]))
        if new_error < error:
            error = new_error

    return error


def estimate_signal_from_data(data, sigma):
    mu_1 = signal_first_moment(data)
    mu_2 = signal_mu_2_from_data(data, sigma)
    P_x = signal_power_spectrum_from_data(data, sigma)
    return estimate_signal(mu_1, mu_2, P_x)


def estimate_signal(mu_1, mu_2, P_x):
    """
    :param mu_1: Estimation of the average of the signal
    :param mu_2: Estimation of the outer product
    :param sigma: standard deviation of our data
    :param P_x: power spectrum vector
    """
    L = len(P_x)
    # Abs not necessary, just to prevent weird output when sigma is high
    P_x = np.abs(P_x)
    v_p = np.sqrt(np.diag(1 / P_x))
    W = (1 / np.sqrt(L)) * scipy.linalg.dft(L)
    Q = W @ v_p @ W.conjugate()

    # Abs not necessary, just to prevent weird output when sigma is high
    mu_2_corrected = np.abs(mu_2)
    mu_2_tilde = Q @ mu_2_corrected @ Q.conjugate()
    eigvals, eigvectors = np.linalg.eig(mu_2_tilde)
    u = eigvectors[:, np.argmax(eigvals)]
    v_tilde = np.fft.ifft(np.fft.fft(u) * np.sqrt(P_x))
    x = ((np.sum(mu_1) / np.sum(v_tilde)) * v_tilde).real
    C_x = np.zeros((L, L))
    for i in range(L):
        C_x[:, i] = x.take(INDEXES[i])
    rho = np.linalg.inv(C_x) @ mu_1
    return x, rho


signal = np.array([1, 2, 3], dtype='float')
print("signal: ", signal)
mean = np.mean(signal)
N = 10000
L = len(signal)

INDEXES = np.ones((L, L), dtype='int')
for i in range(L):
    INDEXES[i] = np.roll(np.arange(L), i)


def experiment(x: np.array, N, sigma):
    data = generate_data(x, N, sigma)
    estimated_signal, rho = estimate_signal_from_data(data, sigma)
    print(f"Experiment: N={N}, x={x}, sigma={sigma}")
    error = np.linalg.norm(estimated_signal - x)
    print(f"Error: {error}")
    print(f"Rho: {rho}\n")

L = 3
INDEXES = np.ones((L, L), dtype='int')
for i in range(L):
    INDEXES[i] = np.roll(np.arange(L), i)


experiment(np.array([1, 2, 3]), N=10000, sigma=1)
experiment(np.array([1, 2, 3]), N=100000, sigma=1)
experiment(np.array([1, 2, 3]), N=1000000, sigma=1)

experiment(np.array([1, 2, 3]), N=10000, sigma=10)
experiment(np.array([1, 2, 3]), N=100000, sigma=10)
experiment(np.array([1, 2, 3]), N=1000000, sigma=10)

L = 10
INDEXES = np.ones((L, L), dtype='int')
for i in range(L):
    INDEXES[i] = np.roll(np.arange(L), i)

experiment(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), N=10000, sigma=1)
experiment(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), N=100000, sigma=1)
experiment(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), N=1000000, sigma=1)
