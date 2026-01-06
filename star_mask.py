from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import numpy as np

# Load the FITS image
fits_file = './examples/test_M31_linear.fits'
#fits_file = './examples/test_M31_raw.fits'
hdul = fits.open(fits_file)

# Print basic information about the FITS file
hdul.info()

# Get image data from the main HDU
data = hdul[0].data

# Read header metadata (not used later, but available if needed)
header = hdul[0].header

# Check if the image is color or monochrome
if data.ndim == 3:
    # Color image case

    # If channels come first (3, height, width), move them to last position
    if data.shape[0] == 3:
        data = np.transpose(data, (1, 2, 0))
    
    # For star detection, use luminance (average of RGB channels)
    data_for_detection = np.mean(data, axis=2)
    
    # Normalize image to [0, 1] for proper display
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Save the original color image
    plt.imsave('./results/original.png', data_normalized)

else:
    # Monochrome image case

    data_for_detection = data
    
    # Save the original grayscale image
    plt.imsave('./results/original.png', data, cmap='gray')

# Compute background statistics for star detection
# These values help separate stars from noise
mean, median, std = sigma_clipped_stats(data_for_detection, sigma=3.0)

print(f"Image statistics: mean={mean:.2f}, median={median:.2f}, std={std:.2f}")

# Create the star finder object
# fwhm: approximate star size in pixels
# threshold: how bright a star has to be compared to background noise
daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)

# Detect stars after removing background
sources = daofind(data_for_detection - median)

print(f"Number of stars detected: {len(sources)}")
print("First few stars:")
print(sources[:5])

# Create an empty mask image
# 0 = background, 255 = star
mask = np.zeros(data_for_detection.shape, dtype=np.uint8)

# Loop over detected stars and draw a circle around each one
for star in sources:
    xc, yc = int(star['xcentroid']), int(star['ycentroid'])
    
    # Fixed radius around each star (in pixels)
    radius = 5
    
    # Build a circular mask
    y, x = np.ogrid[:data_for_detection.shape[0], :data_for_detection.shape[1]]
    distance = np.sqrt((x - xc)**2 + (y - yc)**2)
    mask[distance <= radius] = 255

# Save the star mask
plt.imsave('./results/star_mask.png', mask, cmap='gray')

# Create a figure to visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Show original image
if data.ndim == 3:
    axes[0].imshow(data_normalized)
else:
    axes[0].imshow(data, cmap='gray',
                   vmin=np.percentile(data, 1),
                   vmax=np.percentile(data, 99))
axes[0].set_title('Original image')
axes[0].axis('off')

# Show detected stars on top of the image
if data.ndim == 3:
    axes[1].imshow(data_normalized)
else:
    axes[1].imshow(data, cmap='gray',
                   vmin=np.percentile(data, 1),
                   vmax=np.percentile(data, 99))
axes[1].scatter(
    sources['xcentroid'],
    sources['ycentroid'],
    s=50,
    facecolors='none',
    edgecolors='red',
    linewidths=1
)
axes[1].set_title(f'Detected stars ({len(sources)})')
axes[1].axis('off')

# Show the binary star mask
axes[2].imshow(mask, cmap='gray')
axes[2].set_title('Star mask')
axes[2].axis('off')

# Save the visualization
plt.tight_layout()
plt.savefig('./results/star_detection_visualization.png',
            dpi=150,
            bbox_inches='tight')
plt.close()

print("Star mask created successfully!")
print("Saved files:")
print("  - ./results/star_mask.png")
print("  - ./results/star_detection_visualization.png")

# Close the FITS file
hdul.close()
