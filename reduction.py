import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

# =========================
# 1. LOAD FITS IMAGE
# =========================
fits_file = "./examples/test_M31_linear.fits"
hdul = fits.open(fits_file)
data = hdul[0].data.astype(np.float32)
hdul.close()

if data.ndim == 3 and data.shape[0] == 3:
    data = np.transpose(data, (1, 2, 0))

color = data.ndim == 3

if color:
    R, G, B = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    luminance = (R + G + B) / 3.0
else:
    luminance = data.copy()

# =========================
# 2. STAR DETECTION
# =========================
mean, median, std = sigma_clipped_stats(luminance, sigma=3.0)
daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
sources = daofind(luminance - median)

print(f"{len(sources)} étoiles détectées")

# =========================
# 3. LUMINANCE REDUCTION
# =========================
L_norm = (luminance - luminance.min()) / (luminance.max() - luminance.min())
L_uint8 = (L_norm * 255).astype(np.uint8)
L_reduced = L_uint8.copy()

for star in sources:
    x = int(star["xcentroid"])
    y = int(star["ycentroid"])
    flux = star["flux"]

    diameter = int(np.clip(2.0 * np.sqrt(flux), 3, 25))
    if diameter % 2 == 0:
        diameter += 1

    half = diameter // 2
    y1, y2 = max(0, y - half), min(L_reduced.shape[0], y + half + 1)
    x1, x2 = max(0, x - half), min(L_reduced.shape[1], x + half + 1)

    patch = L_reduced[y1:y2, x1:x2].copy()

    kernel_size = max(3, diameter // 2)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for _ in range(6):
        patch = cv.erode(patch, kernel, iterations=1)
        if np.count_nonzero(patch > 15) <= 1:
            break

    L_reduced[y1:y2, x1:x2] = patch

# =========================
# 4. COLOR REINJECTION
# =========================
L_reduced_f = L_reduced.astype(np.float32) / 255.0
L_orig_f = L_uint8.astype(np.float32) / 255.0

ratio = L_reduced_f / (L_orig_f + 1e-6)

ratio_min = 0.25
ratio = np.clip(ratio, ratio_min, 1.0)

ratio = cv.GaussianBlur(ratio, (5, 5), 0)

if color:
    out = np.zeros_like(data)
    out[:, :, 0] = R * ratio
    out[:, :, 1] = G * ratio
    out[:, :, 2] = B * ratio
else:
    out = luminance * ratio

# =========================
# 5. SAVE RESULT
# =========================
out_norm = (out - out.min()) / (out.max() - out.min())
plt.imsave("./results/reduction_max.png", out_norm)

print("Réduction couleur terminée.")
print("Image générée : ./results/reduction_max.png")