# molecular_spectra_analysis

This python module is computing the average spectra under masked spots of a cube map (Ra-Dec-Vel), fitting the spectra with a Gaussian profile, then displaying them one by one (in .pdf) and writing out fitted parameters (in .dat).

# Dependencies

- numpy
- astropy (fits, WCS, Table)
- scipy (curve_fit)

# Usage

Download C18O (1-0) example FITS images from the ./indir directory; download and run molecular_spectra_analysis.py

# Created and Tested:

Created by Vera Konyves. 
Tested with Python 3.8 in Ipython.

