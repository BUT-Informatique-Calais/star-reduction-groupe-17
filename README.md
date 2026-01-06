# Star Reduction in astrophotology
*Authors*: Lara Fremy, Antoine Lutsen, Antoine Molinaro (TD3)

# Project Documentation
*Presentation*:
This project is a Star Reduction tool for .fits astronomic pictures treatement. The goal is to reduce brillant stars apparent diameter without altering the background nebulous structures

*Problem stateme,t*:
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



