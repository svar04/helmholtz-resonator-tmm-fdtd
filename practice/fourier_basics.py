import numpy as np
import matplotlib.pyplot as plt

sample_rate = 44100
signal_length = 1
d = 1/sample_rate

time = np.linspace(0, signal_length, sample_rate)


f1 = 200
f2 = 400
f3 = 800

signal_1 = np.sin(2 * np.pi * f1 * time)
signal_2 = np.sin(2 * np.pi * f2 * time)
signal_3 = np.sin(2 * np.pi * f3 * time)
signal_sum = signal_1 + signal_2 + signal_3

fft_result = np.fft.fft(signal_sum)
freq = np.fft.fftfreq(sample_rate, 1/sample_rate)[:sample_rate//2]
fft_peaks = np.abs(fft_result)[:sample_rate//2]

plt.figure(str(f1) + " HZ")
plt.plot(time, signal_1)
plt.ylim(-3, 3)
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.figure(str(f2) + " HZ")
plt.plot(time, signal_2)
plt.ylim(-3, 3)
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.figure(str(f3) + " HZ")
plt.plot(time, signal_3)
plt.ylim(-3, 3)
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.figure("Sum of signals")
plt.plot(time, signal_sum)
plt.ylim(-3, 3)
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.figure("Fast Fourier Transform")
plt.plot(freq, fft_peaks)
plt.show()