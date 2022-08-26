# Radio-Forced-Photometry

Forced photometry on a radio intensity image (should work on images at other wavelengths). 

It measures flux densities of sources in units of the intensity map
 - fitting 2D Gaussian,
 - fitting 2D off-axis Lorentzian profiles,
 - optionally using a given aperture radius (in arcseconds).
 
It also reports peak flux, rms value at the source position 
and 1-sigma errors on fluxes obtained from the fitting. 
All measurements will added to a given catalogue as new columns.
There will be also two additonal columns giving comments on values obtained from fitting [C: the source likely overlaps with a bright source which will depend on the flux limit you provide, A: accept].


Requirements:

- a fits catalogue with two columns 'RA' and 'DEC', 
- intensity map as fits image
- noise/rms map as fits image
- 5sigma flux threshold (in Jy based on image data)
- [optional] aperture radius (in arcsec)


Notes: lmfit and various tools/codes have been utilised to compose this code and further plotting utilites are under development.
Feedbacks are welcome.
