import numpy as np
import matplotlib.pyplot as plt

#Setup Variables

l = 0.002       # 2mm neck length
r = 0.006       # 6mm neck side
p = 1.225
c = 343

l = l + 1.7 * r

L_cav = 0.043   # 86mm cavity
S_cav = (0.014**2)*2   # 14x14mm square cavity
V = S_cav * L_cav

Sn = np.pi * r**2

offset = 0.02
l1 = L_cav/2 - offset
l2 = L_cav/2 + offset

f = np.arange(1,6001, 1)
Z = np.zeros(6000, dtype=complex)

#Print out variables

print(l, r, p, c, V, Sn, f, Z)

Zn = p*c/Sn
Z_cav = p*c/S_cav

# Replace the en tire 'for' loop and Z initialization with these three lines:
k = 2 * np.pi * f / c
phi = np.arctan2(Z_cav * np.cos(k*l1) * np.cos(k*l2), Zn * np.sin(k*(l1 + l2)))
Z = 1j * Zn * np.tan(k*l - phi)

print(Z)
print(f)
print(Z.shape)
print(f.shape)

print(f"l1 = {l1}, l2 = {l2}")
print(f"Z at 100Hz = {Z[99]}")
print(f"Z at 500Hz = {Z[499]}")
print(f"Z at 1000Hz = {Z[999]}")

rPipe = 0.1
sPipe = 0.03**2

zPipe = p*c/sPipe

magnitudes = np.abs(1 + (zPipe/(2*Z)))

TL = 20*np.log10(magnitudes)

rFreqIndex = np.argpartition(TL, -2)[-2:]
print(f[rFreqIndex])
print(TL[rFreqIndex])

max_tl_index = np.argmax(TL)
print(f"Single Peak Frequency: {f[max_tl_index]} Hz")
print(f"Max Transmission Loss: {TL[max_tl_index]} dB")

plt.plot(f, TL)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Transmission Loss')
plt.title('Transmission Loss of Different Frequencies')
plt.show()