#!/usr/bin/env python3
import numpy as np
import matplotlib,sys,os
from astropy.io import fits
from astropy import wcs
from astropy.wcs.utils import proj_plane_pixel_scales as ppps
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import lmfit
from lmfit import Parameters,minimize,report_fit,Model
from lmfit.lineshapes import gaussian2d, lorentzian


""" Functions """


def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x,y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def lorentzian2d(x, y, amplitude=1., centerx=0., centery=0., sigmax=1., sigmay=1.,
                 rotation=0):
    """Return a two dimensional lorentzian.

    The maximum of the peak occurs at ``centerx`` and ``centery``
    with widths ``sigmax`` and ``sigmay`` in the x and y directions
    respectively. The peak can be rotated by choosing the value of ``rotation``
    in radians.
    """
    xp = (x - centerx)*np.cos(rotation) - (y - centery)*np.sin(rotation)
    yp = (x - centerx)*np.sin(rotation) + (y - centery)*np.cos(rotation)
    R = (xp/sigmax)**2 + (yp/sigmay)**2

    return 2*amplitude*lorentzian(R)/(np.pi*sigmax*sigmay)


def flattenn(f):
	naxis=f[0].header['NAXIS']
	if naxis<2:
		print ('Can\'t make map from this')
	w = wcs.WCS(f[0].header)
	wn=wcs.WCS(naxis=2)

	wn.wcs.crpix[0]=w.wcs.crpix[0]
	wn.wcs.crpix[1]=w.wcs.crpix[1]
	wn.wcs.cdelt=w.wcs.cdelt[0:2]
	wn.wcs.crval=w.wcs.crval[0:2]
	wn.wcs.ctype[0]=w.wcs.ctype[0]
	wn.wcs.ctype[1]=w.wcs.ctype[1]

	header = wn.to_header()
	header["NAXIS"]=2
	copy=('EQUINOX','EPOCH')
	for k in copy:
	 r=f[0].header.get(k)
	 if r:
	     header[k]=r

	slice=(0,)*(naxis-2)+(np.s_[:],)*2

	return header,f[0].data[slice]

def get_map(filename):
    '''Open FITS image.'''
    rmap = fits.open(filename, ext=1,memmap=True, lazy_load_hdus=False)
    return rmap


def get_beam (radio_map):
    '''Calculate beam size of a radio map.'''
    image=get_map(radio_map)
    bmaj=image[0].header['BMAJ']
    bmin=image[0].header['BMIN']
    w = wcs.WCS(image[0].header)
    # find the pixel size in arcsec
    pix_size = 3600.0*ppps(w)[0] # per pixel in arcsecsonds
    #print ("Image pixel size is ",round(pix_size), "arcsec")
    # find out the beam area
    gfactor=2.0*np.sqrt(2.0*np.log(2.0))
    prhd = image[0].header
    w=wcs.WCS(prhd)
    cd1=-w.wcs.cdelt[0]
    cd2=w.wcs.cdelt[1]
    if ((cd1-cd2)/cd1)>1.0001 and ((bmaj-bmin)/bmin)>1.0001:
        raise RadioError('Pixels are not square (%g, %g) and beam is elliptical' % (cd1, cd2))
    bmaj/=cd1
    bmin/=cd2
    beam_area=2.0*np.pi*(bmaj*bmin)/(gfactor*gfactor)
    #print ('Bmin:',round(bmin),' pixels', 'Bmaj:',round(bmaj),' pixels')
    #print ("Beam area in pixels: ",beam_area)
    return beam_area,pix_size,image[0].header['BMAJ'],image[0].header['BMIN'],image[0].header['BPA']

def point_rms(ra,dec,data,wcs):
    re_coord = [ra,dec]
    im_coord = wcs.wcs_world2pix([re_coord],0)
    x_ra = int(round(im_coord[0][0]))
    x_dec = int(round(im_coord[0][1]))
    noise = data[x_dec,x_ra]
    return noise


def aper_peak_flux(ra,dec,data,aper_p,wcs_im,beam):
    """ aper_p in pixels """
    global flux_ap
    re_coord = [ra,dec]
    im_coord = wcs_im.wcs_world2pix([re_coord],0)
    x_ra = im_coord[0][0]
    x_dec = im_coord[0][1]
    ysize,xsize=data.shape
    if x_ra<aper_p or x_dec<aper_p or x_ra>(xsize-aper_p) or x_dec>(ysize-aper_p):
        flux_ap=noise=peak_flux=np.nan
    else:
        xmin=int(x_ra-(aper_p+1))
        xmax=int(x_ra+(aper_p+1))
        ymin=int(x_dec-(aper_p+1))
        ymax=int(x_dec+(aper_p+1))
        x_pra=x_ra-xmin
        x_pdec=x_dec-ymin
        subim=data[ymin:ymax,xmin:xmax]
        H, W = subim.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        mask = (x - x_pra)**2 + (y - x_pdec)**2 <= aper_p**2
        flux_ap=np.sum(subim[mask])/beam  
        peak_flux=np.max(subim[mask])
    return flux_ap,peak_flux

def gaussian_simple(xsize,ysize,x0,y0,xs,ys,pa):
    X, Y = np.meshgrid(np.arange(0,xsize,1.0), np.arange(0,ysize,1.0))
    pa=-(90.0+pa) # N through E
    pa*=np.pi/180.0
    a=0.5*((np.cos(pa)/xs)**2.0+(np.sin(pa)/ys)**2.0)
    b=0.25*((-np.sin(2*pa)/xs**2.0)+(np.sin(2*pa)/ys**2.0))
    c=0.5*((np.sin(pa)/xs)**2.0+(np.cos(pa)/ys)**2.0)
    #return ne.evaluate('exp(-(a*(X-x0)**2.0+2*b*(X-x0)*(Y-y0)+c*(Y-y0)**2.0))')/(2*np.pi*xs*ys)
    gaussian=np.exp(-(a*(X-x0)**2.0+2*b*(X-x0)*(Y-y0)+c*(Y-y0)**2.0))/(2*np.pi*xs*ys)
    #return np.ravel(gaussian)
    return gaussian


def calc_app_correction(aperture,radiomap):
    image=get_map(radiomap)
    bmaj = image[0].header['BMAJ']*3600.
    pix_size = get_beam(radiomap)[1]
    bmaj=bmaj/pix_size       # pixels
    sigma=bmaj/2.355         # sigma=FWHM/2.355
    aper_p = aperture/pix_size # pixels; aperture is in arcsec
    x_ra=50.1;x_dec=50.66
    data=gaussian_simple(100,100,x_ra,x_dec,sigma,sigma,0)
    xmin=int(x_ra-(aper_p+1))
    xmax=int(x_ra+(aper_p+1))
    ymin=int(x_dec-(aper_p+1))
    ymax=int(x_dec+(aper_p+1))
    x_pra=x_ra-xmin
    x_pdec=x_dec-ymin
    subim=data[ymin:ymax,xmin:xmax]
    H, W = subim.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    mask = (x - x_pra)**2 + (y - x_pdec)**2 <= aper_p**2
    flux=np.sum(subim[mask])
    print ("Aperture correction: ",flux)
    return flux