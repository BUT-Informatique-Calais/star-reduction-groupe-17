import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load previously generated images

original = cv.imread('./results/original.png', cv.IMREAD_UNCHANGED)
eroded = cv.imread('./results/eroded.png', cv.IMREAD_UNCHANGED)
mask = cv.imread('./results/star_mask.png', cv.IMREAD_GRAYSCALE)

# Remove alpha channel if present
if original.shape[2] == 4:
    original = original[:, :, :3]

# Convert to float in [0,1]
original_f = original.astype(np.float32) / 255.0
eroded_f = eroded.astype(np.float32) / 255.0
mask_f = mask.astype(np.float32) / 255.0

# Blur + soften the mask
mask_blurred = cv.GaussianBlur(mask_f, (17, 17), 0)
mask_blurred = np.clip(mask_blurred, 0, 1)
mask_blurred = np.power(mask_blurred, 2.2)

# Adapt mask to 3 channels
if mask_blurred.ndim == 2:
    mask_blurred = np.repeat(mask_blurred[:, :, np.newaxis], 3, axis=2)

# Controlled interpolation
strength = 0.55
final = (mask_blurred * strength) * eroded_f + (1.0 - mask_blurred * strength) * original_f

# Convert to uint8 for saving
final_uint8 = np.clip(final * 255, 0, 255).astype(np.uint8)

# Save the result

cv.imwrite('./results/final.png', final_uint8)

print("Localized reduction completed.")
print("Generated file: ./results/final.png")


# présence Antoine Lutsen
