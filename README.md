# Radio-Forced-Photometry

Forced photometry on a radio intensity image (should work on other images). 

It measures flux densities of sources in units of the intensity map
 - using a given aperture (in arcseconds)
 - fitting 2D Gaussian 
 - fitting 2D off-axis Lorentzian profiles.
 
It also reports peak flux in a given aperture, rms value at the source position 
and 1sigma errors on fluxes obtained from the fitting. 
All new measurements will added to the given catalogue.

Requirements:

- a fits catalogue with two columns RA and DEC, 
- intensity map as fits image
- noise/rms map as fits image
- aperture value (arcsec)
