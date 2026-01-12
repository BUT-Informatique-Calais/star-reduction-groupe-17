# Star Reduction in astrophotography
*Authors*: Lara Fremy, Antoine Lutsen, Antoine Molinaro (TD3)

# Project Documentation
*Presentation*:
This project is a Star Reduction tool for .fits astronomic pictures treatement. The goal is to reduce brillant stars apparent diameter without altering the background nebulous structures

*Problem statement*:
In astrophotography, bright stars are often "bloated" by atmospheric diffusion and PSF (Point Spread Function). When applying histogram stretches to reveal nebula details, these stars can saturate the image and mask important structures.

## Installation


### Virtual Environment

It is recommended to create a virtual environment before installing dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


### Dependencies
```bash
pip install -r requirements.txt
```

Or install dependencies manually:
```bash
pip install [package-name]

pip install astropy photutils opencv-python numpy matplotlib
```

## Usage


### Command Line
```bash
python main.py [arguments]
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Examples files
Example files are located in the `examples/` directory. You can run the scripts with these files to see how they work.
- Example 1 : `examples/HorseHead.fits` (Black and whiteFITS image file for testing)
- Example 2 : `examples/test_M31_linear.fits` (Color FITS image file for testing)
- Example 3 : `examples/test_M31_raw.fits` (Color FITS image file for testing)


# Implemented Methods

# Phase 1: Global Erosion
Using OpenCV's morphological erosion operator to uniformly reduce all bright structures in the image.

It's a simple and fast implementation wich Effectively reduces star size.

But it alters nebula details. We also face a loss of information in low-luminosity areas.

## Phase 2: Selective Detection and Masking

# Step A: Star Mask Creation

#### Script: `star_mask.py`
Star mask uses DAOStarFinder for automatic star identification and creates gradient masks for smooth transitions.

**How does it work?**
- **Statistics calculation**: Mean, median, and standard deviation with outlier removal (sigma_clipped_stats)
- **Source detection**: Identifying brightness peaks exceeding threshold (5σ above background)
- **Gradient mask generation**: Creating smooth radial gradients around each detected star proportional to star brightness
- **Gaussian blur application** for smoother transitions

**Attenuation method:**
Instead of simple darkening, the script blends star regions towards the local background estimate (calculated with large-scale Gaussian filter). This preserves natural appearance.

**Pros:**
- Precise detection based on statistical criteria
- Smooth gradient masks for natural transitions
- Preservation of nebula regions
- Handles both color and monochrome images

**Cons:**
- May miss very faint stars below detection threshold
- Computational cost higher than simple erosion

**Output files:**
- `original.png`: Normalized original image
- `eroded.png`: Star-reduced image
- `star_mask.png`: Visualization of the gradient mask (hot colormap)
- `star_detection_visualization.png`: 4-panel comparison
- `comparison.png`: Side-by-side before/after

---

### Step B: Localized Reduction with Mask Blending

#### Script: `localized_reduction.py`
Combines the original image, eroded image, and star mask to create a selective reduction.

**How it works:**
- Loads three inputs: original image, eroded image, and star mask
- Applies Gaussian blur to soften mask transitions
- Uses gamma correction (power 2.2) to adjust mask strength
- Performs weighted interpolation between original and eroded images based on mask

**Key parameters:**
- `strength = 0.55`: Controls reduction intensity (0 = no effect, 1 = full erosion)
- Gaussian blur kernel (17x17): Ensures smooth transitions
- Gamma 2.2: Makes mask falloff more natural

**Pros:**
- Selective processing preserves nebula details
- Smooth transitions between reduced and original regions
- Adjustable reduction strength

**Cons:**
- Requires three separate input files
- Processing pipeline split across multiple scripts

---

## Phase 3: Advanced Color-Preserving Reduction

### Script: `reduction.py`
Most sophisticated approach combining luminance-based star detection with color-preserving reduction.

**How it works:**

**1. FITS Loading & Color Handling:**
- Loads FITS data and handles channel ordering
- Computes luminance channel for star detection (average of RGB)

**2. Star Detection:**
- Uses DAOStarFinder on luminance channel
- Detects stars above 5σ threshold

**3. Luminance Reduction:**
- Processes only the luminance channel in uint8 space
- For each detected star:
  - Calculates adaptive diameter based on star flux
  - Extracts local patch around star
  - Applies iterative erosion (up to 6 iterations)
  - Stops early if star becomes too dim
- This creates targeted reduction without affecting background

**4. Color Reinjection:**
- Calculates reduction ratio: `ratio = L_reduced / L_original`
- Applies ratio to each RGB channel independently
- Preserves original color relationships
- Uses Gaussian blur on ratio for smooth transitions
- Clips ratio to minimum 0.25 to avoid over-darkening

**Pros:**
- Preserves color balance and hue information
- Adaptive star size calculation based on brightness
- Targeted processing with minimal nebula impact
- Single-script solution (no intermediate files needed)
- Smart iterative erosion that stops when appropriate

**Cons:**
- More complex algorithm
- Requires understanding of luminance/chrominance separation
- Processing time longer than simple methods

**Output:** `reduction_max.png` with maximum quality color-preserving reduction

---

## Visualization Tool

### Script: `compare.py`
Interactive OpenCV viewer for comparing original and processed images with blink comparison mode.

**Features:**
- Three display modes:
  - Mode 1: Original image
  - Mode 2: Final result
  - Mode 3: Blink animation (alternates between original and final)
- Semi-transparent menu overlay showing current mode
- Automatic image resizing if dimensions don't match
- Keyboard controls:
  - `1`: Show original
  - `2`: Show final result
  - `3`: Enable blink mode
  - `ESC`: Exit viewer

**Usage:**
```bash
python compare.py
```

**Requirements:** 
- `./results/original.png` must exist
- `./results/reduction_max.png` must exist (run `reduction.py` first)

**Pros:**
- Quick visual comparison
- Blink mode helps identify subtle differences
- User-friendly interface

---

## Phase 4: Interactive GUI Application

### Script: `user_interface.py`
Full-featured graphical interface built with Tkinter for real-time star reduction with live preview.

### Main Features:

**1. File Management:**
- FITS file loading with drag-and-drop support
- Save results in multiple formats (PNG, JPEG)
- Automatic image format detection (color/monochrome)
- Real-time status feedback

**2. Star Detection Controls:**
- **Auto-Detection Mode:** Automatically estimates optimal FWHM and detection threshold
  - Uses background subtraction with large-scale Gaussian filter (sigma=20)
  - Performs permissive initial detection to estimate star size
  - Sweeps threshold range to find optimal detection balance (20-3000 stars)
  - Updates FWHM and threshold parameters automatically
- **Manual Mode:** 
  - FWHM slider (1.0-10.0 px): Expected star size for detection
  - Threshold slider (1.0-15.0 σ): Detection sensitivity

**3. Reduction Settings:**
- **Erosion Iterations (1-12):** Number of erosion passes per star
- **Ratio Min (0.0-1.0):** Minimum reduction ratio to prevent over-darkening
- **Multi-Size Reduction:** Advanced feature for differential star treatment
  - Small Stars erosion (1-8): Light erosion for faint stars
  - Medium Stars erosion (3-10): Moderate erosion for average stars
  - Large Stars erosion (5-15): Strong erosion for bright stars
  - Small/Medium Threshold (0.1-0.5): Flux percentile boundary
  - Medium/Large Threshold (0.5-0.9): Flux percentile boundary
  - Automatic star categorization based on flux distribution

**4. Display Adjustments:**
- **Black Point (0-10%):** Sets lower percentile for clipping dark values
- **Stretch/Gamma (0.1-3.0):** Non-linear stretch to reveal faint details
  - Values < 1.0: Brighten faint structures
  - Values > 1.0: Darken and increase contrast
- Real-time display updates (no reprocessing needed)

**5. Comparison Modes:**
- **Original:** View unprocessed FITS data
- **Result:** View star-reduced image
- **Side by Side:** Split-screen before/after comparison with labels
- **Blink:** Animated toggle between original and result (500ms interval)
- Visual mode indicator in top-right corner

**6. Performance Optimizations:**
- **Source Caching:** Stars detected once, reused for parameter adjustments
- **Delayed Updates:** 300ms debounce on slider changes to avoid lag
- **Display-Only Updates:** Stretch and black point don't trigger reprocessing
- **Smart Invalidation:** Detection parameter changes force recalculation

### How the Processing Works:

**1. Image Loading:**
- Opens FITS file and extracts float32 data
- Handles channel transposition for color images
- Calculates luminance for star detection

**2. Auto-Calibration (if enabled):**
- Subtracts large-scale background (sigma=20 Gaussian)
- Runs permissive detection (threshold=2.0σ, FWHM=3.0)
- Estimates median FWHM from detected sources
- Sweeps thresholds from 1.5σ to 4.0σ
- Selects threshold yielding 20-3000 stars

**3. Star Detection:**
- Uses DAOStarFinder with calibrated or manual parameters
- Detects stars above background threshold
- Caches results to avoid recomputation

**4. Luminance Reduction:**
- Normalizes luminance to uint8 using 1st-99.5th percentiles
- For each detected star:
  - Calculates adaptive diameter from flux
  - Extracts local patch around star
  - Applies erosion with kernel size = diameter/2
  - Uses differential erosion if multi-size enabled
  - Stops early if star becomes too dim
- Creates reduction ratio map

**5. Color Preservation:**
- Calculates ratio = L_reduced / L_original
- Clips ratio to [ratio_min, 1.0]
- Applies Gaussian blur (5x5) for smooth transitions
- Multiplies each RGB channel by ratio independently
- Preserves original color relationships

**6. Display Normalization:**
- Applies black point clipping (removes background noise)
- Stretches with gamma curve for visual enhancement
- Scales to [0, 255] for display



### Keyboard/Mouse Controls:
- Mouse wheel: Scroll parameter panel
- Slider drag: Adjust parameters in real-time
- Click view buttons: Switch comparison modes
- Window resize: Auto-scales image display

### Pros:
- Real-time interactive preview
- No command-line knowledge required
- Visual feedback for all parameters
- Multiple comparison modes for quality assessment
- Auto-detection eliminates guesswork
- Multi-size reduction for optimal star handling
- Efficient caching and delayed updates

### Cons:
- Requires GUI environmen
- Memory intensive for large images

---

## Results Organization

All processed images are saved in the `./results/` directory:

- `original.png`: Original FITS data normalized to PNG
- `eroded.png`: Global erosion result
- `dilated.png`: Global dilation result (from dilate.py)
- `star_mask.png`: Gradient mask visualization
- `star_detection_visualization.png`: 4-panel diagnostic view
- `comparison.png`: Side-by-side before/after
- `final.png`: Localized reduction result
- `reduction_max.png`: Final color-preserving reduction

---

## Recommended Workflow

For best results, follow this processing order:

### 1. Test Phase 1 (Optional):
```bash
python erosion.py
```
Understand limitations of global erosion

### 2. Phase 2 - Generate Mask:
```bash
python star_mask.py
```
Creates detection masks and visualizations

### 3. Phase 2 - Apply Localized Reduction:
```bash
python localized_reduction.py
```
Combines mask with erosion

### 4. Phase 3 - Best Quality (Recommended):
```bash
python reduction.py
```
One-step color-preserving reduction

### 5. Compare Results:
```bash
python compare.py
```
Interactive visualization

### 6. GUI Interface (Interactive):
```bash
python user_interface.py
```
Real-time parameter adjustment with live preview

---

## Technical Notes

### FITS File Handling
The scripts automatically handle:
- Monochrome images (2D arrays)
- Color images (3D arrays)
- Different channel orderings: (3, H, W) → (H, W, 3)
- Float32 to uint8 conversion for OpenCV
- Normalization preserving color relationships

### Star Detection Parameters
Adjustable in `DAOStarFinder`:
- `fwhm=3.0`: Full Width Half Maximum (expected star size)
- `threshold=5.0 * std`: Detection sensitivity (5-sigma)

### Tuning Reduction Strength

**In `star_mask.py`:**
- `attenuation = 0.6`: Star reduction amount (0-1)
- `sigma=20`: Background estimation blur radius

**In `localized_reduction.py`:**
- `strength = 0.55`: Blend factor between original and eroded

**In `reduction.py`:**
- `ratio_min = 0.25`: Minimum reduction ratio (prevents over-darkening)
- Erosion iterations: Adaptive per star (max 6)

---

## Troubleshooting

**"Aucune étoile détectée" (No stars detected):**
- Lower the threshold multiplier in DAOStarFinder (try 3.0 instead of 5.0)
- Check if FITS file loaded correctly
- Verify image has sufficient dynamic range

**Color artifacts in reduced images:**
- Increase Gaussian blur sigma in color reinjection
- Adjust ratio_min value in reduction.py
- Check that ratio clipping is active

**Nebula details lost:**
- Reduce attenuation/strength parameters
- Increase background estimation sigma
- Use reduction.py instead of erosion.py

**File not found errors:**
- Verify FITS file path exists
- Create `./results/` directory if missing
- Run scripts in correct order (star_mask.py before localized_reduction.py)


## Credits & References

This project uses:
- **Astropy**: FITS file handling and statistics
- **Photutils**: DAOStarFinder algorithm
- **OpenCV**: Morphological operations and image processing
- **NumPy/SciPy**: Numerical operations and Gaussian filtering
- **Matplotlib**: Image saving and visualization
- **Tkinter**: GUI interface
- **Pillow (PIL)**: Image display in GUI

---

## License

This project is developed as part of academic coursework (SAE ASTROPHOTO).
