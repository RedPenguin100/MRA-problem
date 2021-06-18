import numpy as np


def generate_data(x: np.array, N, sigma):
    """
    :param x: signal
    :param N: measurements amount
    :param sigma: standard deviation
    :return:
    """
    L = x.size
    group_members = np.random.randint(0, L, N)
    noise = np.random.normal(scale=sigma, size=(N, L))
    data = np.zeros((N, L))

    for i in range(N):
        data[i] = np.roll(x, group_members[i]) + noise[i]

    return data


def signal_to_noise_ratio(x, sigma):
    return np.power(np.norm(x) / sigma)


def signal_mean_from_data(data):
    """
    :note: The mean of each row separately, and then
    averaging the results, is the same result as averaging
    everything together.
    """
    return np.mean(data)


def signal_power_spectrum_from_data(data):
    global sigma
    global L
    power_spectra = np.power(np.abs(np.fft.fft(data)), 2.0) - np.power(sigma, 2.0) * L
    return np.mean(power_spectra, axis=0)


arr = np.array([1, 2, 3, 4, 5, 6])
mean = np.mean(arr)
N = 1000000
L = len(arr)
sigma = 0
real_ps = signal_power_spectrum_from_data(arr.reshape((1, 6)))
print(real_ps)
sigma = 2
data = generate_data(arr, N, sigma)

print(signal_power_spectrum_from_data(data[:10]) - real_ps)
print(signal_power_spectrum_from_data(data[:100]) - real_ps)
print(signal_power_spectrum_from_data(data[:1000]) - real_ps)
print(signal_power_spectrum_from_data(data[:10000]) - real_ps)
print(signal_power_spectrum_from_data(data[:100000]) - real_ps)
print(signal_power_spectrum_from_data(data[:1000000]) - real_ps)
# errors = np.zeros(N)
#
# for i in range(N // 20, len(errors)):
#     errors[i] = np.linalg.norm(signal_mean_from_data(data[:i + 1]) - mean)
#
# plt.plot(range(N), errors, 'r--')
# plt.show()
