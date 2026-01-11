from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Load the FITS image
fits_file = "./examples/test_M31_linear.fits"
hdul = fits.open(fits_file)

# Get image data from the main HDU
data = hdul[0].data.astype(np.float32)

# Close FITS file
hdul.close()

# Check if the image is color or monochrome
if data.ndim == 3:
    # Color image case
    if data.shape[0] == 3:
        data = np.transpose(data, (1, 2, 0))
    
    # For star detection, use luminance (average of RGB channels)
    luminance = np.mean(data, axis=2)
else:
    # Monochrome image case
    luminance = data.copy()

# Compute background statistics for star detection
mean, median, std = sigma_clipped_stats(luminance, sigma=3.0)

print(f"Image statistics: mean={mean:.2f}, median={median:.2f}, std={std:.2f}")

# Create the star finder object
daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)

# Detect stars after removing background
sources = daofind(luminance - median)

if sources is None:
    print("Aucune étoile détectée")
    sources = []
else:
    print(f"Number of stars detected: {len(sources)}")


# Create an empty star mask with smooth gradients
mask = np.zeros(luminance.shape, dtype=np.float32)

# Loop over detected stars and create a smooth gradient mask
y_grid, x_grid = np.ogrid[:luminance.shape[0], :luminance.shape[1]]

for star in sources:
    xc, yc = star["xcentroid"], star["ycentroid"]
    
    # Star intensity (brightness)
    star_flux = star["flux"]
    
    # Radius proportional to star brightness (but with limits)
    radius = np.clip(3.0 + star_flux / 10000.0, 3.0, 10.0)
    
    # Calculate distance from star center
    distance = np.sqrt((x_grid - xc)**2 + (y_grid - yc)**2)
    
    # Create smooth radial gradient (1 at center, 0 at radius)
    star_mask = np.maximum(0, 1.0 - distance / radius)
    
    # Use maximum to handle overlapping stars
    mask = np.maximum(mask, star_mask)

# Apply Gaussian blur to make the mask even smoother
mask = gaussian_filter(mask, sigma=1.5)

# Star reduction strength (0 = no effect, 1 = full removal)
attenuation = 0.6

# Calculate background level for each position
# This is the local background around each star
background_estimate = gaussian_filter(luminance, sigma=20)

# Apply the star mask intelligently
if data.ndim == 3:
    # For color images
    data_reduced = data.copy()
    
    # Calculate how much to reduce each pixel
    # Instead of darkening, we blend towards the local background
    for i in range(3):
        channel = data[:, :, i]
        bg_channel = gaussian_filter(channel, sigma=20)
        
        # Blend star regions towards background
        data_reduced[:, :, i] = channel * (1.0 - mask * attenuation) + bg_channel * mask * attenuation * 0.3
else:
    # For monochrome images
    bg = gaussian_filter(data, sigma=20)
    data_reduced = data * (1.0 - mask * attenuation) + bg * mask * attenuation * 0.3

# Normalize function that preserves color relationships
def normalize_preserve_color(img):
    """Normalize image while preserving color balance"""
    if img.ndim == 3:
        # For color images: normalize each channel independently
        # but use percentiles from the luminance
        lum = np.mean(img, axis=2)
        p1, p99 = np.percentile(lum[lum > 0], (1, 99))
        
        img_norm = np.zeros_like(img)
        for i in range(3):
            channel = img[:, :, i]
            img_norm[:, :, i] = np.clip((channel - p1) / (p99 - p1), 0, 1)
    else:
        # For grayscale images
        p1, p99 = np.percentile(img[img > 0], (1, 99))
        img_norm = np.clip((img - p1) / (p99 - p1), 0, 1)
    
    return img_norm

# Save results
plt.imsave("./results/original.png", normalize_preserve_color(data))
plt.imsave("./results/eroded.png", normalize_preserve_color(data_reduced))
plt.imsave("./results/star_mask.png", mask, cmap="hot")

# Visualization: original + detected stars + eroded + mask
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

axes[0, 0].imshow(normalize_preserve_color(data))
axes[0, 0].set_title("Image originale")
axes[0, 0].axis("off")

axes[0, 1].imshow(normalize_preserve_color(data))
if len(sources) > 0:
    axes[0, 1].scatter(
        sources["xcentroid"],
        sources["ycentroid"],
        s=40,
        facecolors="none",
        edgecolors="green",
        linewidths=0.8
    )

axes[0, 1].set_title(f"Étoiles détectées ({len(sources)})")
axes[0, 1].axis("off")

axes[1, 0].imshow(normalize_preserve_color(data_reduced))
axes[1, 0].set_title("Étoiles réduites")
axes[1, 0].axis("off")

axes[1, 1].imshow(mask, cmap="hot")
axes[1, 1].set_title("Masque des étoiles (gradient)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig(
    "./results/star_detection_visualization.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# Create a comparison image
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(normalize_preserve_color(data))
axes[0].set_title("Avant (original)", fontsize=14, weight='bold')
axes[0].axis("off")

axes[1].imshow(normalize_preserve_color(data_reduced))
axes[1].set_title("Après (étoiles réduites)", fontsize=14, weight='bold')
axes[1].axis("off")

plt.tight_layout()
plt.savefig(
    "./results/comparison.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("Réduction des étoiles terminée avec succès!")
print("Fichiers sauvegardés:")
print("  - ./results/original.png")
print("  - ./results/eroded.png")
print("  - ./results/star_mask.png")
print("  - ./results/star_detection_visualization.png")
print("  - ./results/comparison.png")