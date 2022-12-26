import imageio
from skimage import transform
from skimage.color import rgb2gray
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import numpy as np

# ---
# Signal from HW05
# 1. DFT by using built-in FFT function
f = np.array([1,0,1,0],dtype=np.float64)
F = np.fft.fft(f)
print(F)

# 2. DFT by manually calculating according to definition of DFT
M = 4
Fm = np.zeros((M,),dtype=complex)
for u in range(M):
	Fm[u] = np.sum(np.exp(-1j*2*np.pi*u*x/M) * x for x in f)
print(Fm)

# ---
# Image DFT
im = imageio.imread('MaruTaro.jpg')
im = rgb2gray(im)
### imageio.imwrite('MaruTaro_grayscale.png', im)
plt.imshow(im, cmap='gray')
plt.show()


z = np.fft.fft2(im)
print('z[0,0] = {:f}, mean(im) = {:f}'.format(z[0,0], np.sum(im)))

# We skip directly to drawing the spectrum with fftshift applied.
zs = np.fft.fftshift(z)
mag = np.absolute(zs)
ang = np.angle(zs)

plt.subplot(121)
plt.imshow(np.log(np.absolute(z)), cmap='gray')
plt.title('Before shifting:')
plt.subplot(122)
plt.imshow(np.log(np.absolute(zs)), cmap='gray')
plt.title('After shifting:')
plt.show()

plt.subplot(121)
plt.imshow(np.log(mag), cmap='gray')
plt.title('Magnitude')
plt.subplot(122)
plt.imshow(ang, cmap='gray')
plt.title('Phase angle')
plt.show()

# ---
# Halftone
im2 = imageio.imread('halftone2.jpg')
im2 = rgb2gray(im2)
z2 = np.fft.fftshift(np.fft.fft2(im2))
mag2 = np.absolute(z2)
ang2 = np.angle(z2)

plt.subplot(121)
plt.imshow(np.log(mag2), cmap='gray')
plt.title('Magnitude')
plt.subplot(122)
plt.imshow(ang2, cmap='gray')
plt.title('Phase angle')
plt.show()

# ---
# Ideal Low pass filtering without padding
# 1
height, width = im.shape
P = height
Q = width
# 2
F = np.fft.fftshift(np.fft.fft2(im))
# 3
D0 = 60.0
U, V = np.meshgrid(np.arange(P), np.arange(Q), indexing='ij')
H = (U-P/2)**2 + (V-Q/2)**2 <= D0**2 # compare squared distance to squared D0 instead
# 4
G = H * F
# 5
gp = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
# 6
g = gp[:height, :width]

# imageio.imwrite('LP_60.png', H)
# imageio.imwrite('MaruTaro_LP_60.png', g)

plt.subplot(121)
plt.imshow(H, cmap='gray')
plt.title('Ideal Lowpass Filter')
plt.subplot(122)
plt.imshow(g, cmap='gray')
plt.title('Lowpass filtered image')
plt.show()

# ---
# Ideal Low pass filtering with padding
# 1
height, width = im.shape
P = 2*height
Q = 2*width
# 2
# See textbook Ch. 4, pp. 253-266 for more discussion on padding! 
im_padded = np.pad(im, ((0, P-height), (0, Q-width)), mode='constant', constant_values=(0,0))
# 3
F = np.fft.fftshift(np.fft.fft2(im_padded))
# 4
D0 = 60.0
U, V = np.meshgrid(np.arange(P), np.arange(Q), indexing='ij')
H = (U-P/2)**2 + (V-Q/2)**2 <= D0**2 # compare squared distance to squared D0 instead
# 5
G = H * F
# 6
gp = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
# 7
g = gp[:height, :width]

plt.subplot(121)
plt.imshow(H, cmap='gray')
plt.title('Ideal Lowpass Filter')
plt.subplot(122)
plt.imshow(g, cmap='gray')
plt.title('Lowpass filtered image')
plt.show()
