# Created by Vera Konyves  

################
# Imports, setup
################

import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS
from astropy.table import Table
from astropy.io import fits
from scipy.optimize import curve_fit
print('Imports done!')

indir = './indir/'
outdir = './outdir/'

# Definition of Gaussian function
def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2./(2.*sigma**2.))
    
# Read in C18O fits file (in main beam temperature units, Ra-Dec-Vel dimensions)
in_Tmb = indir+'example_C18O_Tmb.fits'

hdu = fits.open(in_Tmb)[0]
data_cube, hdr_cube = hdu.data, hdu.header

# Cube Header
wcs = WCS(hdr_cube)
naxis = hdr_cube["NAXIS"]
naxis1 = hdr_cube["NAXIS1"]
naxis2 = hdr_cube["NAXIS2"]
cdelt1 = hdr_cube["CDELT1"]
naxis3 = hdr_cube["NAXIS3"]
crval3 = hdr_cube["CRVAL3"]
crpix3 = hdr_cube["CRPIX3"]
cdelt3 = hdr_cube["CDELT3"]

# Velocity axis
vaxis = np.array([i for i in range(naxis3)])
vaxis = crval3 + (vaxis - crpix3 + 1) * cdelt3
vaxis = vaxis/1000.  #m/s to km/s
print("C18O cube image read in!")

# rms noise level of moment0 map is expected to be RMS_per_chan/channel_number^0.5
RMS_per_chan = 0.20	# rms per channel
#chan_num = 70. - 22. + 1.	# check spectra, from-to channel numbers, ADJUST!
#sigma = RMS_per_chan / chan_num**0.5	# ie, RMS_expected, ~0.028, ADJUST!   

# Threshold
rms_thresh = 6.	# 6 sigma

# Find spectra peak between min & max velocities (in km/s)
Vpeak_min = 6.0
Vpeak_max = 12.0


##################################################
# Compute average spectra under the masks, and fit 
##################################################

in_mask = indir+'example_C18O_mom0_masks.fits'   

hdu = fits.open(in_mask)[0]
result = hdu.data


spec_tmp = np.array([0.0 for i in range(naxis3)])
map_tmp = np.zeros([naxis2, naxis1], float)
Tpeak = np.array([0.0 for i in range(np.nanmax(result)+1)]) 
Vsys = np.array([0.0 for i in range(np.nanmax(result)+1)]) 
dV = np.array([0.0 for i in range(np.nanmax(result)+1)])   # Linewidth 
errTpeak = np.array([0.0 for i in range(np.nanmax(result)+1)]) 
errVsys = np.array([0.0 for i in range(np.nanmax(result)+1)]) 
errdV = np.array([0.0 for i in range(np.nanmax(result)+1)])   # Err Linewidth 
RSS = np.array([0.0 for i in range(np.nanmax(result)+1)])   # sum of squared residuals 

for struc in range(np.nanmax(result)+1):
  count = 0.
  spec_tmp = np.array([0.0 for i in range(naxis3)])
  for j in range(naxis2):
    for i in range(naxis1):
      tmp = int(result[j,i]*1.0)
      if int(tmp) == int(struc*1.0):
        count = count + 1.
        for k in range(naxis3):
          if data_cube[k,j,i] >= -100. and data_cube[k,j,i] <= 100.:
            spec_tmp[k] = spec_tmp[k]  + data_cube[k,j,i]

  spot_no = ['1', '2', '3']
       
  spec_tmp = spec_tmp /count 
  max_spec = np.nanmax(spec_tmp)
  max_velo = vaxis[np.argmax(spec_tmp)]
  rms_for_ave_spec = RMS_per_chan/(count)**0.5	 
  p0_init = [max_spec, max_velo, 0.3]	# Initial assumpition for linewidth: 0.3 km/s 
  if max_velo <= Vpeak_min or max_velo >= Vpeak_max : 
    max_velo = 9.0	# Initial assumpition for Vsys = 9 km/s good for Orion B, ADJUST!
    p0_init = [max_spec, max_velo, 0.3]	# for [amp, cen, wid]
  popt, pcov = curve_fit(gauss_function, vaxis, spec_tmp, p0=p0_init, maxfev = 100000)
  fit_result = gauss_function(vaxis, *popt)    
  Tpeak[struc] = popt[0]
  Vsys[struc] = popt[1]
  dV[struc] = popt[2] * np.sqrt(8 * np.log(2))	# from std sigma to dV: dV = sigma * 2.35482
  perr = np.sqrt(np.diag(pcov))	# 1 stddev errors
  errTpeak[struc] = perr[0]
  errVsys[struc] = perr[1]
  errdV[struc] = perr[2] * np.sqrt(8 * np.log(2))      
  indEmiss = np.where( np.logical_and(vaxis >= Vpeak_min, vaxis <= Vpeak_max) )
  RSS[struc] = np.sum((spec_tmp[indEmiss] - fit_result[indEmiss])**2)	# Residual sum of squares

  print("=====================================")      
  print("Spot No: ", spot_no[struc])
  print("No of pixels in mask: ", count)
  print("Gaussian fitting results:")
  print(" => Tpeak ", Tpeak[struc], " K")
  print(" => Vsys ", Vsys[struc], " km/s")
  print(" => dV ", dV[struc], " km/s")
  print(" => std err Tpeak ", errTpeak[struc], " K")
  print(" => std err Vsys ", errVsys[struc], " km/s")
  print(" => std err dV ", errdV[struc], " km/s")  
  print("")
  print(" => rms ave ", rms_for_ave_spec)
  print(" => RSS ", RSS[struc])


  #########################################
  # Make spectra+fit plot per analysis spot
  ########################################

  fig = plt.figure(figsize=(8, 5))  
  ax = plt.subplot()
  plt.xlabel("velocity [km/s]", fontsize=16)
  plt.ylabel(r"$T_{\rm MB}$ [K]", fontsize=16)
  
  plt.plot(vaxis, spec_tmp, c='dodgerblue', linewidth=6.)
  plt.plot(vaxis, spec_tmp*0., "-", c="black", linewidth=2.)
  plt.plot(vaxis, fit_result, c='red', linewidth=3.)
  plt.plot(vaxis, spec_tmp*0.+rms_for_ave_spec*rms_thresh, "--", c="orange", linewidth=2.)
  plt.axvline(x=Vsys[struc], linewidth=2., color='red', ls='--')

  xmin, xmax = ax.get_xlim()
  ymin, ymax = ax.get_ylim()
  plt.xlim([5, 13])
  plt.ylim([-0.1, ymax])
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  
  plt.text(0.45*xmax, 0.90*ymax, r'C$^{18}$O(1$-$0)', fontsize=26, ha ='center', va ='center', color='dodgerblue')  
  plt.text(0.88*xmax, 0.90*ymax, str(spot_no[struc]), fontsize=26, ha ='center', va ='center', color='k')  


  plt.subplots_adjust(top=0.975, bottom=0.13, left=0.12, right=0.975, hspace=0, wspace=0)
  
  #plt.show(block=False) 
  #plt.close('all')
  plt.savefig(outdir+'C18O_molec_spectra_spotNo'+str(spot_no[struc])+'.pdf')	
  print('Figures saved out!')

###########################################
# Write out dat file with fitted parameters
##########################################

datname_out = 'C18O_molec_spectra_analys.dat'
tab_rows = []
for i in range(np.nanmax(result)+1):
    tab_rows.append(( spot_no[i], round(Tpeak[i], 3), round(errTpeak[i], 3), round(Vsys[i], 3), round(errVsys[i], 3), round(dV[i], 3), round(errdV[i], 3), round(RSS[i], 6) ))

tab = Table(rows=tab_rows, names=['SNo', 'Tpeak_K', 'errTpeak', 'Vsys_kms', 'errVsys', 'dV_kms', 'errdV', 'RSS'])
tab.write(outdir+datname_out, format = 'ascii', overwrite=True)
print('Data file written out!')
#
# SNo: Spot numbers 1, 2, 3, marked by mask values 0, 1, 2, resp., in the mask image (masking out value: -1). 
# Tpeak_K: peak (main beam) temperature of the fit, in K  
# Vsys_kms: systemic velocity at the fitted peak, in km/s 
# dV_kms: FWHM linewidth, in km/s
# errTpeak, errVsys, errdV: 1-standard deviation errors of the above 
# RSS: residual sum of squares calculated from the fits




