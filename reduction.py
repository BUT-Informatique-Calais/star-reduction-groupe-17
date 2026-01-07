import numpy as np
import cv2
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

def magnitude_to_kernel(mag):
    if mag < 8:
        return 9
    elif mag < 10:
        return 7
    elif mag < 12:
        return 5
    else:
        return 3

def reduce_stars_multiscale(image):
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)

    finder = DAOStarFinder(fwhm=3.0, threshold=5. * std)
    stars = finder(image - median)

    if stars is None:
        return image.copy()

    result = image.copy()

    for star in stars:
        x = int(star['xcentroid'])
        y = int(star['ycentroid'])
        flux = star['flux']

        if flux <= 0:
            continue

        magnitude = -2.5 * np.log10(flux)
        k = magnitude_to_kernel(magnitude)

        r = k * 2
        y1, y2 = max(0, y - r), min(image.shape[0], y + r)
        x1, x2 = max(0, x - r), min(image.shape[1], x + r)

        roi = result[y1:y2, x1:x2]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        eroded = cv2.erode(roi, kernel)

        result[y1:y2, x1:x2] = eroded

    return result
