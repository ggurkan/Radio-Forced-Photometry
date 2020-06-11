import sys
sys.path.append('/home/SCRIPTS/')
from utils import *
import matplotlib.pyplot as plt
from astropy.modeling.fitting import LevMarLSQFitter
from math import *
from lmfit import Parameters,minimize,report_fit,Model
from astropy.coordinates import Angle
import skimage
from skimage.draw import (polygon,circle_perimeter,ellipse_perimeter)
from skimage.draw import line_aa, circle_perimeter_aa,ellipse  
gaussian_fwhm_to_sigma = 0.4246609001440095

path = "/beegfs/lofar/deepfields/data_release/en1/"
#lofarcat = Table.read(path+"pybdsf_source.fits") #if there is no cross-matced-processed cat
lofarcat = Table.read(path+"final_cross_match_catalogue-v0.6.fits") 
lcat = SkyCoord(lofarcat['RA'],lofarcat['DEC'],unit="deg")

catalogue = "/smp2/ggu/elaisn1/EN1_opt_sedfit_ds_positions.fits"
cat = Table.read(catalogue) 
RA = np.array(cat['RA'])
DEC = np.array(cat['DEC'])

image = path+"radio_image.fits"
rms = path+"radio_rms_image.fits"
flux_map=get_map(image);fmap=flatten(flux_map)
rms_map=get_map(rms);rmap=flatten(rms_map)
hdu=fmap[0];wcs_im = wcs.WCS(fmap[0])
rhdu=fmap[0];wcs_rms = wcs.WCS(rmap[0])

beam = get_beam(image)[0]                 # beam area in arcsec
pixel_size = get_beam(image)[1]           # size in arcsec per pixel
aper_a=8.                                 # aperture radius in arcsec
appcorr=calc_app_correction(aper_a,image) # aperture correction

aper_a=aper_a/pixel_size

FWHM_x=FWHM_y=4.
x_stddev=FWHM_x*gaussian_fwhm_to_sigma
y_stddev=FWHM_y*gaussian_fwhm_to_sigma;pa=0.

import time
start_time = time.time()

fit_flux=[];rfit=[];flag=[];rmsval=[]
# flag -1: point flux, 0: aperture flux, 1: fit flux 2: cat measurement
for it,i in enumerate(cat):
	print (it)
	""" check if the source is in the catalogue using unique ID name"""
	if cat['ID'][it] in lofarcat['ID']:
		id=cat['ID'][it]
		fit_flux.append(float(lofarcat[lofarcat['ID']==id]['Total_flux']))
		rmsval.append(float(lofarcat[lofarcat['ID']==id]['E_Total_flux']))
		rfit.append(-99.)
		flag.append(int(2))
	else:
		position = SkyCoord(RA[it], DEC[it], unit="deg")
		source = SkyCoord([RA[it]], [DEC[it]], unit="deg")
		idxc0, idxcatalog0, d2d0, d3d0 = lcat.search_around_sky(source, 30.*u.arcsec)
		#idxc, idxcatalog, d2d, d3d = lcat.search_around_sky(source, 2.*u.arcsec)
		if len(d2d0)>=1 and np.all(d2d0.value*3600.<=30.):
			overlaps=[]
			for p in idxcatalog0:
				reco = [lofarcat[p]['RA'], lofarcat[p]['DEC']]
				imco = wcs_im.wcs_world2pix([reco],0)
				majo=lofarcat[p]['Maj']*3600./pixel_size
				mino=lofarcat[p]['Min']*3600./pixel_size
				x0, y0 = imco[0][0],imco[0][1]
				sp=wcs_im.wcs_world2pix([RA[it]], [DEC[it]],0)
				e = skimage.draw.ellipse(x0,y0, majo, mino, shape=None, rotation=np.deg2rad(lofarcat[p]['PA']))
				s = skimage.draw.circle(sp[0][0],sp[1][0], aper_a, shape=None)
				overlaps.append(len(np.intersect1d(e,s)))
			overlaps=np.array(overlaps)
			if np.any(overlaps>=3):
				""" check: aperture overlaps with another source at least 3 pixels"""	
				noise = point_rms(RA[it], DEC[it],rmap[1],wcs_rms)
				rmsval.append(noise)
				flux = point_flux(RA[it], DEC[it],fmap[1],aper_a,wcs_im)
				fit_flux.append(flux)
				rfit.append(-99.)
				flag.append(int(-1))
			else:
				cutout_img = Cutout2D(fmap[1], position, (11, 11), wcs=wcs_im)
				subim=cutout_img.data
				ysize,xsize=subim.shape
				im_mesh=np.meshgrid(np.arange(0,xsize,1.0), np.arange(0,ysize,1.0))
				x0,y0=cutout_img.position_cutout[0],cutout_img.position_cutout[1]
				guess_vals = [1., x0, y0, x_stddev,y_stddev]				
				model = Model(gaussian2d)
				result = model.fit(np.ravel(subim), 
	                               im_mesh=im_mesh, 
	                               amp=guess_vals[0], 
	                               x0=guess_vals[1], 
	                               y0=guess_vals[2], 
	                               x_stddev=guess_vals[3],  
	                               y_stddev=guess_vals[4])
				Rsquared = 1 - result.residual.var()/np.var(subim)

				if Rsquared<=0.1 or list(result.best_values.values())[0]/beam<-0.5 or list(result.best_values.values())[0]/beam>0.5:
					cutout_img0 = Cutout2D(fmap[1], position, (9, 9), wcs=wcs_im)
					subim0=cutout_img0.data
					ysize,xsize=subim0.shape
					im_mesh=np.meshgrid(np.arange(0,xsize,1.0), np.arange(0,ysize,1.0))
					x0,y0=cutout_img0.position_cutout[0],cutout_img0.position_cutout[1]
					guess_vals = [1., x0, y0, x_stddev,y_stddev]				
					model = Model(gaussian2d)
					result0 = model.fit(np.ravel(subim0), 
	                               		im_mesh=im_mesh, 
	                               		amp=guess_vals[0], 
	                               		x0=guess_vals[1], 
	                               		y0=guess_vals[2], 
	                               		x_stddev=guess_vals[3], 
	                               		y_stddev=guess_vals[4])
					Rsquared0 = 1 - result0.residual.var()/np.var(subim0)
					if Rsquared0<=0.1 or list(result0.best_values.values())[0]/beam<-0.5 or list(result0.best_values.values())[0]/beam>0.5:
						H, W = subim.shape
						x, y = np.meshgrid(np.arange(W), np.arange(H))
						x_ra=cutout_img.position_cutout[0];x_dec=cutout_img.position_cutout[1]
						mask = (x - x_ra)**2 + (y - x_dec)**2 <= aper_a**2
						flux_ap=np.sum(subim[mask]/beam)/appcorr
						fit_flux.append(flux_ap)
						rfit.append(-99.)
						flag.append(int(0))
						noise = point_rms(RA[it], DEC[it],rmap[1],wcs_rms)
						rmsval.append(noise)
					else:
						fit_flux.append(list(result0.best_values.values())[0]/beam)
						rfit.append(Rsquared0)
						flag.append(int(1))
						noise = point_rms(RA[it], DEC[it],rmap[1],wcs_rms)
						rmsval.append(noise)

				else:
					fit_flux.append(list(result.best_values.values())[0]/beam)
					rfit.append(Rsquared)
					flag.append(int(1))
					noise = point_rms(RA[it], DEC[it],rmap[1],wcs_rms)
					rmsval.append(noise)
	
		else:
			cutout_img = Cutout2D(fmap[1], position, (11, 11), wcs=wcs_im)
			subim=cutout_img.data
			ysize,xsize=subim.shape
			im_mesh=np.meshgrid(np.arange(0,xsize,1.0), np.arange(0,ysize,1.0))
			x0,y0=cutout_img.position_cutout[0],cutout_img.position_cutout[1]
			guess_vals = [1., x0, y0, x_stddev,y_stddev]				
			model = Model(gaussian2d)
			result = model.fit(np.ravel(subim), 
                	           im_mesh=im_mesh, 
                        	   amp=guess_vals[0], 
                        	   x0=guess_vals[1], 
                        	   y0=guess_vals[2], 
                        	   x_stddev=guess_vals[3], 
                        	   y_stddev=guess_vals[4])
			Rsquared = 1 - result.residual.var()/np.var(subim)
			if Rsquared<=0.1 or list(result.best_values.values())[0]/beam<-0.5 or list(result.best_values.values())[0]/beam>0.5:
				H, W = subim.shape
				x, y = np.meshgrid(np.arange(W), np.arange(H))
				x_ra=cutout_img.position_cutout[0];x_dec=cutout_img.position_cutout[1]
				mask = (x - x_ra)**2 + (y - x_dec)**2 <= aper_a**2
				flux_ap=np.sum(subim[mask]/beam)/appcorr
				fit_flux.append(flux_ap)
				rfit.append(-99.)
				flag.append(int(0))
				noise = point_rms(RA[it], DEC[it],rmap[1],wcs_rms)
				rmsval.append(noise)
			
			else:
				fit_flux.append(list(result.best_values.values())[0]/beam)
				rfit.append(Rsquared)
				flag.append(int(1))
				noise = point_rms(RA[it], DEC[it],rmap[1],wcs_rms)
				rmsval.append(noise)

print (len(fit_flux),len(rfit),len(flag),len(rmsval))
print("--- %s seconds ---" % (time.time() - start_time))	
cat['fitflux']= np.array(fit_flux)
cat['rfit']   = np.array(rfit)
cat['fflag']= np.array(flag)
cat['RMS']    = np.array(rmsval)
cat.write('/smp2/ggu/elaisn1/EN1_opt_sedfit_ds_positions.fits',overwrite=True)
