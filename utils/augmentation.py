import numpy as np


def jitter(data, sigma=0.03):
    noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
    return data + noise


def scaling(data, sigma=0.1):
    factor = np.random.normal(loc=1.0, scale=sigma)
    return data * factor


def time_warp(data, sigma=0.2):
    warp_factor = np.random.normal(loc=1.0, scale=sigma)
    return np.interp(np.arange(0, len(data), warp_factor), np.arange(0, len(data)), data)


def permutation(data, max_segments=5):
    permuted_data = data.copy()
    orig_segments = np.random.randint(1, max_segments)
    segments = np.split(np.arange(len(data)), orig_segments)
    np.random.shuffle(segments)
    return np.concatenate([permuted_data[s] for s in segments])


def window_slice(data, slice_ratio=0.9):
    slice_size = int(len(data) * slice_ratio)
    start_idx = np.random.randint(0, len(data) - slice_size)
    return data[start_idx:start_idx + slice_size]
