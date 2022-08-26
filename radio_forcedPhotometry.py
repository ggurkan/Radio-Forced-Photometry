#!/usr/bin/env python3
from fputils import *
import argparse

aperflux=[];err_aperflux=[]
pflux=[];err_pflux=[]
flux2d=[];err_flux2d=[]
lorenflux=[];err_lorenflux=[]
comgaus=[];comloren=[]

def main(incatname, intmap, noisemap, flxlim, aperture):

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

	bm_area,pixel_size,bm_maj,bm_min,bm_pa=get_beam(intmap)
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

	flxlim=flxlim*10.

	if args.aperture:
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
				extend = extend+3 #add 2.5 pixels on each side
			else:
				extend = extend # add 2 pixel on each side

			extend = extend*pixel_size
			size   = u.Quantity((extend, extend), u.arcsec)

			for i,it in enumerate(cat):
				ra  = rain[i]
				dec = decin[i]
				
				#aperture flux and noise from the rms map

				apflux   = aper_peak_flux(ra,dec,idata,aperture/pixel_size,wcs_im,bm_area)[0]/appcorr
				noise    = point_rms(ra,dec,rmsdata,wcs_rms)
				peakf    = aper_peak_flux(ra,dec,idata,aperture/pixel_size,wcs_im,bm_area)[1]

				position = SkyCoord(ra, dec, unit="deg")
				cutout   = Cutout2D(idata, position, size, wcs=wcs_im)
				
				#only keep data value of inner region that has the source
				a,b = cutout.data.shape
				cutout.data[0:2,0:a]=0.;cutout.data[a-2:a,0:a]=0.
				cutout.data[2:a-2,0:2]=0.;cutout.data[2:a-2,a-2:a]=0.

				# get cutout to estimate flux by fitting Gaussian2D and Lorentzian2D profile
				xd,yd = cutout.data.shape
				x,y   = np.meshgrid(np.arange(yd), np.arange(xd))		
				
				#2D Gaussian fitting
				model_2d   = lmfit.models.Gaussian2dModel()
				params_2d  = model_2d.guess(np.ravel(cutout.data),np.ravel(x),np.ravel(y))
				result_2d  = model_2d.fit(np.ravel(cutout.data),x=np.ravel(x),y=np.ravel(y), params=params_2d)

				gaus2dflux = result_2d.best_values['amplitude']/bm_area

				if gaus2dflux>=flxlim:
					print ('Overlapping with a bright source?')
					comgaus.append('C')
				elif gaus2dflux<0:
					gaus2dflux    = apflux
					gaus2dfluxerr = noise
					comgaus.append('A')
				else:
					comgaus.append('A')

				if result_2d.params['amplitude'].stderr is not None:
					gaus2dfluxerr = result_2d.params['amplitude'].stderr/bm_area
				else:
					gaus2dfluxerr = 0.

				#Off-axis 2D Lorentzian profile fitting
				model_lorentz  = lmfit.Model(lorentzian2d, independent_vars=['x', 'y'])
				params_lorentz = model_lorentz.make_params(amplitude=10, centerx=np.ravel(x)[np.argmax(np.ravel(cutout.data))],
				                            centery=np.ravel(y)[np.argmax(np.ravel(cutout.data))])
				params_lorentz['rotation'].set(value=.1, min=0, max=np.pi/2)
				params_lorentz['sigmax'].set(value=1, min=0)
				params_lorentz['sigmay'].set(value=2, min=0)

				result_lorentz = model_lorentz.fit(np.ravel(cutout.data),x=np.ravel(x),y=np.ravel(y), params=params_lorentz)
				loren2dflux    = result_lorentz.best_values['amplitude']/bm_area
				if loren2dflux>=flxlim:
					print ('Overlapping with a bright source?')
					comloren.append('C')
				elif loren2dflux<0:
					loren2dflux    = apflux
					loren2dfluxerr = noise
					comloren.append('A')
				else:
					comloren.append('A')	

				if result_lorentz.params['amplitude'].stderr is not None:
					loren2dfluxerr = result_lorentz.params['amplitude'].stderr/bm_area
				else:
					loren2dfluxerr = 0.
				

				aperflux.append(apflux)
				pflux.append(peakf)
				err_aperflux.append(noise)
				flux2d.append(gaus2dflux)
				err_flux2d.append(gaus2dfluxerr)
				lorenflux.append(loren2dflux)
				err_lorenflux.append(loren2dfluxerr)


			cat['aperflux']        = aperflux
			cat['aperflux_err']    = err_aperflux
			cat['peakflux']        = pflux
			cat['flux2dGaus']      = flux2d
			cat['flux2dGaus_err']  = err_flux2d
			cat['Comment_Gaus']    = comgaus
			cat['flux2dLoren']     = lorenflux
			cat['flux2dLoren_err'] = err_lorenflux
			cat['Comment_Loren']   = comloren

			cat.write(incatname, overwrite=True)
			return 0

	else:
		""" Estimate size of a cutout based on the map's resolution """
		resolution = np.sqrt(bm_maj*bm_min*3600**2)
		aperture   = resolution + 3.
		aper_ext   = (aperture/pixel_size)*2
		extend     = aper_ext+4
		if extend % 2 == 0:
			extend = extend+3 #add 2.5 pixels on each side
		else:
			extend = extend # add 2 pixel on each side

		extend = extend*pixel_size
		size   = u.Quantity((extend, extend), u.arcsec)

		for i,it in enumerate(cat):
			ra  = rain[i]
			dec = decin[i]
			
			#peak flux from the map and noise from the rms map
			peakf    = aper_peak_flux(ra,dec,idata,aperture/pixel_size,wcs_im,bm_area)[1]
			noise    = point_rms(ra,dec,rmsdata,wcs_rms)

			position = SkyCoord(ra, dec, unit="deg")
			cutout   = Cutout2D(idata, position, size, wcs=wcs_im)
			
			#only keep data value of inner region that has the source
			a,b = cutout.data.shape
			cutout.data[0:2,0:a]=0.;cutout.data[a-2:a,0:a]=0.
			cutout.data[2:a-2,0:2]=0.;cutout.data[2:a-2,a-2:a]=0.

			# get cutout to estimate flux by fitting Gaussian2D and Lorentzian2D profile
			xd,yd = cutout.data.shape
			x,y   = np.meshgrid(np.arange(yd), np.arange(xd))		
			
			#2D Gaussian fitting
			model_2d  = lmfit.models.Gaussian2dModel()
			params_2d = model_2d.guess(np.ravel(cutout.data),np.ravel(x),np.ravel(y))
			result_2d = model_2d.fit(np.ravel(cutout.data),x=np.ravel(x),y=np.ravel(y), params=params_2d)

			gaus2dflux=result_2d.best_values['amplitude']/bm_area
			if gaus2dflux>=flxlim:
				print ('Overlapping with a bright source?')
				comgaus.append('C')
			elif gaus2dflux<0:
				apflux        = aper_peak_flux(ra,dec,idata,aperture/pixel_size,wcs_im,bm_area)[0]/appcorr
				noise         = point_rms(ra,dec,rmsdata,wcs_rms)
				gaus2dflux    = apflux
				gaus2dfluxerr = noise
				comgaus.append('A')
			else:
				comgaus.append('A')

			if result_2d.params['amplitude'].stderr is not None:
				gaus2dfluxerr = result_2d.params['amplitude'].stderr/bm_area
			else:
				gaus2dfluxerr = 0.

			#Off-axis 2D Lorentzian profile fitting
			model_lorentz  = lmfit.Model(lorentzian2d, independent_vars=['x', 'y'])
			params_lorentz = model_lorentz.make_params(amplitude=10, centerx=np.ravel(x)[np.argmax(np.ravel(cutout.data))],
			                            centery=np.ravel(y)[np.argmax(np.ravel(cutout.data))])
			params_lorentz['rotation'].set(value=.1, min=0, max=np.pi/2)
			params_lorentz['sigmax'].set(value=1, min=0)
			params_lorentz['sigmay'].set(value=2, min=0)

			result_lorentz = model_lorentz.fit(np.ravel(cutout.data),x=np.ravel(x),y=np.ravel(y), params=params_lorentz)
			loren2dflux    = result_lorentz.best_values['amplitude']/bm_area
			if loren2dflux>=flxlim:
				print ('Overlapping with a bright source?')
				comloren.append('C')
			elif loren2dflux<0:
				apflux         = aper_peak_flux(ra,dec,idata,aperture/pixel_size,wcs_im,bm_area)[0]/appcorr
				noise          = point_rms(ra,dec,rmsdata,wcs_rms)
				loren2dfluxerr = noise
				comloren.append('A')	
			else:
				comloren.append('A')

			if result_lorentz.params['amplitude'].stderr is not None:
				loren2dfluxerr = result_lorentz.params['amplitude'].stderr/bm_area
			else:
				loren2dfluxerr = 0.
	
			pflux.append(peakf)
			err_pflux.append(noise)
			flux2d.append(gaus2dflux)
			err_flux2d.append(gaus2dfluxerr)
			lorenflux.append(loren2dflux)
			err_lorenflux.append(loren2dfluxerr)



		cat['peakflux']        = pflux
		cat['peakflux_err']    = err_pflux
		cat['flux2dGaus']      = flux2d
		cat['flux2dGaus_err']  = err_flux2d
		cat['Comment_Gaus']    = comgaus
		cat['flux2dLoren']     = lorenflux
		cat['flux2dLoren_err'] = err_lorenflux
		cat['Comment_Loren']   = comloren

		cat.write(incatname, overwrite=True)
		return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('catalogue',               type=str,    help = 'Name of input catalogue including path')
    parser.add_argument('intmap',                  type=str,    help = 'Name of intensity map including path')
    parser.add_argument('noisemap',                type=str,    help = 'Name of noise map including path')
    parser.add_argument('flxlim',                  type=float,  help = '5-sigma minimum flux expected (Jy)')    
    parser.add_argument('--aperture',  default=0,  type=int,    help = 'Aperture radius in arcsec')
    args = parser.parse_args()
       
    main(args.catalogue, args.intmap, args.noisemap, args.flxlim, args.aperture)

