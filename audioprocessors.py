from scipy.fft import fft, ifft
import numpy as np
import pywt

# Basic Stuff
def do_nothing(data : bytes):
    temp = bytes_to_int(data)
    print(temp)
    return int_to_bytes(temp)

def microphone(data : bytes):
    temp = bytes_to_int(data)
    temp *= 2
    return int_to_bytes(confine_ints(temp))

"""
Adds Gaussian noise to a signal. Mean of 0, stddev of 30
"""
def noise(data : bytes, rng : np.random.Generator):
    temp = bytes_to_int(data)
    noise = rng.normal(0, 30, temp.shape)
    return int_to_bytes(confine_ints(temp + noise))

# Wavelet Stuff
def do_wavelet_for_fun(data : bytes):
    temp = bytes_to_int(data)
    coeffs = pywt.wavedec(temp, 'haar',level=5)
    retVal = pywt.waverec(coeffs, 'haar')
    return int_to_bytes(confine_ints(retVal))


def wavelet_low_pass_filter(data : bytes):
    levels = 4
    temp = bytes_to_int(data)
    coeffs = pywt.wavedec(temp, 'db6',level=levels)

    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    for i in range(3, levels + 1):
        arr[coeff_slices[i]['d']] = [0] * len(arr[coeff_slices[i]['d']])
    coeffs = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec')

    retVal = pywt.waverec(coeffs, 'db6')
    return int_to_bytes(confine_ints(retVal))

def wavelet_high_pass_filter(data : bytes):
    levels = 5
    temp = bytes_to_int(data)
    coeffs = pywt.wavedec(temp, wavelet='haar',level=levels)

    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    arr[coeff_slices[0]] = [0] * len(arr[coeff_slices[0]])
    # for i in range(1, 2):
    #     arr[coeff_slices[i]['d']] = [0] * len(arr[coeff_slices[i]['d']])
    coeffs = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec')

    retVal = pywt.waverec(coeffs, wavelet='haar')
    return int_to_bytes(confine_ints(retVal))

def wavelet_threshold_filter(data : bytes, threshold : float):
    levels = 6
    temp = bytes_to_int(data)
    coeffs = pywt.wavedec(temp, 'haar',level=levels)

    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    for i in range(1, 4):
        arr[coeff_slices[i]['d']] = pywt.threshold(arr[coeff_slices[i]['d']], threshold, mode='soft')
    coeffs = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec')

    retVal = pywt.waverec(coeffs, 'haar')
    return int_to_bytes(confine_ints(retVal))

# Fourier Stuff
def do_ft_for_fun(data : bytes):
    temp = bytes_to_int(data)
    y = fft(temp)
    yinv = ifft(y)
    return int_to_bytes(confine_ints(yinv))

def low_pass_filter(data : bytes):
    temp = bytes_to_int(data)
    y = fft(temp)

    cutoff = int(len(y) * 0.25)
    # Delete top 25% and bottom 25% of array
    y = np.concatenate([np.zeros(cutoff - 1), y[cutoff - 1:-cutoff], np.zeros(cutoff)])

    yinv = ifft(y)
    return int_to_bytes(confine_ints(yinv))

def high_pass_filter(data : bytes):
    temp = bytes_to_int(data)
    y = fft(temp)

    arr_size = len(y)
    cutoff = int(arr_size * 0.25)
    # Delete middle 50% of array
    y = np.concatenate([y[0:cutoff], np.zeros(arr_size-2*cutoff),y[-cutoff:]])

    yinv = ifft(y)
    return int_to_bytes(confine_ints(yinv))


# Helper Functions
# https://stackoverflow.com/questions/72034769/what-format-and-data-type-is-data-in-the-stream-read-in-pyaudio
def bytes_to_int(audio : bytes) -> np.ndarray:
    return np.fromstring(audio,dtype=np.int16)

def int_to_bytes(data : np.ndarray) -> bytes:
    return data.tobytes()

def confine_ints(data : np.ndarray) -> np.ndarray:
    return np.round(data).astype(np.int16)