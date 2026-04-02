import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

l = 1
r = 1
p = 1.225
c = 343
V = 3

Sn = np.pi * r**2
lEq = l + 2 * 0.85 * r


f = np.arange(1,6001, 1)
Z = np.zeros(6000, dtype=complex)

print(l, r, p, c, V, Sn, lEq, f, Z)

for i in range(0,6000):

    w = 2 * np.pi * f[i]

    Z[i] = 1j*p *(w * lEq/Sn - c**2/ (w*V) )
    print(Z)

print(Z.shape)
print(f.shape)

print(f)

rPipe = 0.1
sPipe = np.pi * rPipe**2

zPipe = p*c/sPipe

magnitudes = np.abs(1 + (zPipe/(2*Z)))

TL = 20*np.log10(magnitudes)

rFreqIndex = np.argmax(TL)
print(f[rFreqIndex])


plt.plot(f, TL)
plt.show()
