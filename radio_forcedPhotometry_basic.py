#!/usr/bin/env python3
from fputils import *
import argparse


# a basic version that does forced photometry using an aperture or fitting a single gausssian or giving just a positional flux

"""
#example run: python3 radio_forcedPhotometry.py catalogue.fits mosaic.fits rms-map.fits --flux_method [--aperture aperture_value]

"""

aperflux=[];err_aperflux=[]
pflux=[];err_pflux=[]
flux2d=[];err_flux2d=[]
posflux=[];err_posflux=[]

def main(incatname, intmap, noisemap, flux_method, aperture):

	""" Load the catalogue in fits format which must have RA and DEC columns """
	cat     = Table.read(incatname, format='fits')
	rain    = cat['RA']
	decin   = cat['DEC']

	""" Essential information about the intensity map """
	imap    = get_map(intmap)
	imap    = flattenn(imap)
	ihdu    = imap[0]
	wcs_im  = wcs.WCS(ihdu)
	idata   = imap[1]

	bm_area,pixel_size,bm_maj,bm_min=get_beam(intmap)
	print ('...')
	print ('Resolution of the map:',round(bm_maj*3600.,3),'by',round(bm_min*3600.,3),'arcsec')
	print ("Image pixel size is",round(pixel_size,3), "arcsec")
	print ('...\n')

	""" Essential information about the noise map """
	rmsmap  = get_map(noisemap)
	rmsmap  = flattenn(rmsmap)
	rmshdu  = rmsmap[0]
	wcs_rms = wcs.WCS(rmshdu)
	rmsdata = rmsmap[1]


	if flux_method == 'aperture':

		if aperture <= np.sqrt(bm_maj*bm_min*3600**2):
			print ("Aperture is too small! ("+str(aperture),'arcsec) is equal/smaller than the beam ('+str((round(np.sqrt(bm_maj*bm_min*3600**2)))),"arcsec)")
			print ('!!!\n')
			sys.exit()

		else:
			""" Aperture correction """
			appcorr = calc_app_correction(aperture,intmap)

			""" Estimate size of a cutout """
			aper_ext = (aperture/pixel_size)*2
			extend   = aper_ext+4
			if extend % 2 == 0:
				extend = extend+3 # add 2.5 pixels on each side
			else:
				extend = extend # add 2 pixel on each side

			extend = extend*pixel_size
			size   = u.Quantity((extend, extend), u.arcsec)

			for i,it in enumerate(cat):
				ra  = rain[i]
				dec = decin[i]
				print (ra,dec)

				#aperture flux and noise from the rms map
				apflux   = aper_flux(ra,dec,idata,aperture/pixel_size,wcs_im,bm_area)/appcorr
				noise    = point_rms(ra,dec,rmsdata,wcs_rms)		

				aperflux.append(apflux)
				err_aperflux.append(noise)

			cat['aperflux']        = aperflux
			cat['aperflux_err']    = err_aperflux

			cat.write(incatname, overwrite=True)
			return 0

	elif flux_method == 'positional':

		for i,it in enumerate(cat):
			ra  = rain[i]
			dec = decin[i]
			print (ra,dec)

			positionflux   = position_flux(ra,dec,idata,wcs_im)
			err_positionflux = point_rms(ra,dec,rmsdata,wcs_rms)

			posflux.append(positionflux)
			err_posflux.append(err_positionflux)

		cat['positionflux']        = posflux
		cat['positionflux_err']    = err_posflux

		cat.write(incatname, overwrite=True)
		return 0
	

	elif flux_method == 'gaussian_fit':

		for i,it in enumerate(cat):
			ra  = rain[i]
			dec = decin[i]

			""" Estimate size of a cutout """
			# take the resolution of the map to decide the aperture for extend
			aperture = (bm_maj*3600.)+5
			aper_ext = (aperture/pixel_size)*2
			extend   = aper_ext+4
			if extend % 2 == 0:
				extend = extend+3 #add 2.5 pixels on each side
			else:
				extend = extend # add 2 pixel on each side

			extend = extend*pixel_size
			size   = u.Quantity((extend, extend), u.arcsec)


			position = SkyCoord(ra, dec, unit="deg")
			cutout   = Cutout2D(idata, position, size, wcs=wcs_im)
			
			#only keep data value of inner region that has the source
			a,b = cutout.data.shape
			cutout.data[0:2,0:a]=0.;cutout.data[a-2:a,0:a]=0.
			cutout.data[2:a-2,0:2]=0.;cutout.data[2:a-2,a-2:a]=0.

			# get cutout to estimate flux by fitting Gaussian2D 
			xd,yd = cutout.data.shape
			x,y   = np.meshgrid(np.arange(yd), np.arange(xd))		

			#2D Gaussian fitting
			model_2d   = lmfit.models.Gaussian2dModel()
			params_2d  = model_2d.guess(np.ravel(cutout.data),np.ravel(x),np.ravel(y))
			result_2d  = model_2d.fit(np.ravel(cutout.data),x=np.ravel(x),y=np.ravel(y), params=params_2d)

			gaus2dflux = result_2d.best_values['amplitude']/bm_area

			if result_2d.params['amplitude'].stderr is not None:
				gaus2dfluxerr = result_2d.params['amplitude'].stderr/bm_area
			else:
				gaus2dfluxerr = 0.

			flux2d.append(gaus2dflux)
			err_flux2d.append(gaus2dfluxerr)

		cat['flux2dGaus']      = flux2d
		cat['flux2dGaus_err']  = err_flux2d

		cat.write(incatname, overwrite=True)
		return 0
		
	else:
		raise ValueError(f"Unknown flux measurement method: {flux_method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('catalogue',               type=str,    help = 'Name of input catalogue including path')
    parser.add_argument('intmap',                  type=str,    help = 'Name of intensity map including path')
    parser.add_argument('noisemap',                type=str,    help = 'Name of noise map including path')
    parser.add_argument('--aperture',  default=0,  type=int,    help = 'Aperture radius in arcsec')
    parser.add_argument('--flux_method', default='positional', type=str, choices=['aperture', 'gaussian_fit', 'positional'], 
                    help = 'Method to calculate flux: aperture, gaussian_fit, or positional')

    args = parser.parse_args()
       
    main(args.catalogue, args.intmap, args.noisemap, args.flux_method, args.aperture)

