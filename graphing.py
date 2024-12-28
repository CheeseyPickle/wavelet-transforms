import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import pywt
import wave
from PIL import Image

def read_wav(filename : str):
    wav = wave.open(filename, "rb")

    channel = 0
    # Credit: https://stackoverflow.com/questions/51275725/how-do-you-separate-each-channel-of-a-two-channel-wav-file-into-two-different-fi
    nch   = wav.getnchannels()
    depth = wav.getsampwidth()
    wav.setpos(0)
    sdata = wav.readframes(wav.getnframes())

    # Extract channel data (24-bit data not supported)
    typ = { 1: np.int8, 2: np.int16, 4: np.int32 }.get(depth)
    if not typ:
        raise ValueError("sample width {} not supported".format(depth))
    if channel >= nch:
        raise ValueError("cannot extract channel {} out of {}".format(channel+1, nch))
    print ("Extracting channel {} out of {} channels, {}-bit depth".format(channel+1, nch, depth*8))
    data = np.frombuffer(sdata, dtype=typ)
    ch_data = data[channel::nch]

    wav.close()
    return ch_data

def read_img(filename : str):
    image = Image.open(filename)
    pixels = list(image.getdata(0)) # Only get red band from img, since it's in color
    return pixels[0:1920] # The image I'm using is 1920x1080 pixels

if __name__ in "__main__":
    # Read signal from .wav file
    # signal = read_wav("audio/HelloBetter.wav")
    # samples = np.arange(len(signal))

    # plt.plot(samples, signal, label="Original Signal")

    # Read signal from .jpg file
    signal = read_img("image/Selfie.jpg")
    samples = np.arange(len(signal))
    plt.plot(samples, signal, label="Original Signal")

    # # Generate a sine wave
    # freq = 10
    # samples = np.linspace(0, 2.5, 500, endpoint=False)
    # signal = np.sin(2*np.pi*freq*samples)
    
    # # Add noise, if you want
    # rng = np.random.default_rng()
    # noise = rng.normal(0, 0.3, signal.shape)
    # signal = signal + noise
    # plt.plot(samples, signal, label="Original Signal")


    # # Wavelet Transform Time!
    # levels = 1
    # wavelet='haar'
    # coeffs = pywt.wavedec(signal, wavelet=wavelet, level=levels)
    # arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    # # Modify
    # for i in range(1, levels + 1):
    #     arr[coeff_slices[i]['d']] = pywt.threshold(arr[coeff_slices[i]['d']], 0.4, mode='hard')
    # coeffs = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec')
    # retVal = pywt.waverec(coeffs, wavelet=wavelet)

    # plt.plot(samples, retVal, label="Wavelet Filtered")

    # # # Fourier Transform here!
    # y = fft(signal)
    # y = pywt.threshold(y, 50, mode='hard')
    # yinv = ifft(y)

    # plt.plot(samples, yinv, label="FT Filtered")

    # plt.legend()
    # plt.show()

    
    # Wavelet Transform (for compression)
    levels = 3
    wavelet='haar' # can also try bior4.4 for image-compression specific wavelet
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=levels)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    # Modify
    # Delete 3 levels, i.e. 7/8 of the data
    for i in range(1, levels + 1):
        arr[coeff_slices[i]['d']] = [0] * len(arr[coeff_slices[i]['d']])
    coeffs = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec')
    retVal = pywt.waverec(coeffs, wavelet=wavelet)

    plt.plot(samples, retVal, label="Wavelet Compressed")

    # Fourier Compression
    y = fft(signal)
    arr_size = len(y)
    cutoff = int(arr_size * 0.0625)
    # Delete middle 7/8 of transform, i.e. high frequency stuff
    y = np.concatenate([y[0:cutoff], np.zeros(arr_size-2*cutoff),y[-cutoff:]])
    yinv = ifft(y)

    plt.plot(samples, yinv, label="FT Compressed")

    plt.legend()
    plt.show()