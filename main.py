import numpy as np
import pyaudio
import audioprocessors
import wave

SAMPLE_RATE = 44100
SAMPLE_FORMAT = pyaudio.paInt16
CHANNELS = 1
DURATION = 512 # This is in samples

if __name__ in "__main__":
    # Heavily based on PyAudio's documentation
    # https://people.csail.mit.edu/hubert/pyaudio/docs/
    p = pyaudio.PyAudio()

    # If instream is from microphone
    outstream = p.open(format = SAMPLE_FORMAT,
                channels = CHANNELS,
                rate = SAMPLE_RATE,
                output = True)
    instream = p.open(format=SAMPLE_FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                frames_per_buffer=DURATION,
                input=True)
    
    # If instream is from file. Make sure to check the sample rate
    # Also change all the reads to readframes
    # instream = wave.open("audio/HelloConversation.wav", "rb")
    # outstream = p.open(format = SAMPLE_FORMAT,
    #             channels = instream.getnchannels(),
    #             rate = instream.getframerate(),
    #             output = True)
    
    # If adding noise
    rng = np.random.default_rng()

    print("Recording start!")
    data = instream.read(DURATION)

    # Starts an infinite loop. Use Ctrl-C to terminate
    try:
        while (True):
            outstream.write(data)
            data = instream.read(DURATION)
            data = audioprocessors.noise(data, rng)
            data = audioprocessors.wavelet_threshold_filter(data,100)
    
    # Cleanup stuff
    except KeyboardInterrupt:
        print("Interrupted")
        outstream.close()
        instream.close()
        p.terminate()
    