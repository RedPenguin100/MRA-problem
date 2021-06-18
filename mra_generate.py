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
    noise = np.random.normal(scale=np.power(sigma, 2.0), size=(N, L))
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
    pass


sigma = 1
arr = np.array([14123, 11123, -23213, 55551, 96044])
mean = np.mean(arr)
data = generate_data(arr, 100000, sigma)
print(signal_mean_from_data(data[:10]) - mean)
print(signal_mean_from_data(data[:100]) - mean)
print(signal_mean_from_data(data[:1000]) - mean)
print(signal_mean_from_data(data[:10000]) - mean)
print(signal_mean_from_data(data[:100000]) - mean)
