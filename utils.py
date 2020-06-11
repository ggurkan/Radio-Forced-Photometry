import sys
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy import wcs
from astropy.wcs import WCS
from scipy.ndimage.interpolation import map_coordinates, shift
import scipy.optimize as opt
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import Cutout2D
from astropy.modeling.functional_models import Gaussian2D
from astropy.modeling import fitting
from astropy.wcs.utils import proj_plane_pixel_scales as ppps
from matplotlib import cm
from matplotlib import rc, rcParams
import warnings
warnings.filterwarnings("ignore")
#Make use of TeX
rc('text',usetex=True)
# Change all fonts to 'Comptuer Modern'
rc('font',**{'family':'serif','serif':['Times'],'size':12})

gaussian_fwhm_to_sigma = 0.4246609001440095
gaussian_sigma_to_fwhm = 2.3548200450309493 #(FWHM=SIGMA*2.35)

def flatten(f):
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

def gaussian_simple(xsize,ysize,x0,y0,xs,ys,pa):
    X, Y = np.meshgrid(np.arange(0,xsize,1.0), np.arange(0,ysize,1.0))
    pa=-(90.0+pa) # N through E
    pa*=np.pi/180.0
    a=0.5*((np.cos(pa)/xs)**2.0+(np.sin(pa)/ys)**2.0)
    b=0.25*((-np.sin(2*pa)/xs**2.0)+(np.sin(2*pa)/ys**2.0))
    c=0.5*((np.sin(pa)/xs)**2.0+(np.cos(pa)/ys)**2.0)
    #return ne.evaluate('exp(-(a*(X-x0)**2.0+2*b*(X-x0)*(Y-y0)+c*(Y-y0)**2.0))')/(2*np.pi*xs*ys)
    gaussian=np.exp(-(a*(X-x0)**2.0+2*b*(X-x0)*(Y-y0)+c*(Y-y0)**2.0))/(2*np.pi*xs*ys)
    return gaussian

def gaussian(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    '''Fit a two-dimensional Gaussian to the data.'''
    x, y = xy
    a = ((np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2))
    b = (-(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2))
    c = ((np.sin(theta) ** 2)/(2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2))
    g = (offset + amplitude * np.exp( -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) **2 ))))
    g_ravel = g.ravel()
    return g_ravel

def gaussian2d(im_mesh,amp,x0,y0,x_stddev,y_stddev):
    X, Y = im_mesh
    #X, Y = np.meshgrid(np.arange(0,xsize,1.0), np.arange(0,ysize,1.0))
    #pa=-(90.0+pa) # N through E
    #pa*=np.pi/180.0
    pa=0.
    a=0.5*((np.cos(pa)/x_stddev)**2.0+(np.sin(pa)/y_stddev)**2.0)
    b=0.25*((-np.sin(2*pa)/x_stddev**2.0)+(np.sin(2*pa)/y_stddev**2.0))
    c=0.5*((np.sin(pa)/x_stddev)**2.0+(np.cos(pa)/y_stddev)**2.0)
    #return ne.evaluate('exp(-(a*(X-x0)**2.0+2*b*(X-x0)*(Y-y0)+c*(Y-y0)**2.0))')/(2*np.pi*sx*sy)
    gaussian=amp*np.exp(-(a*(X-x0)**2.0+2*b*(X-x0)*(Y-y0)+c*(Y-y0)**2.0))/(2*np.pi*x_stddev*y_stddev)
    return np.ravel(gaussian)

def mossat2d(x, y, amp, x0, y0, gamma, alpha):
    '''2D Mossat model function'''
    func = ((x - x0) ** 2 + (y - y0) ** 2) / gamma ** 2
    return amp * (1 + func) ** (-alpha)


def fit_2DGaussian(data, x0, y0, FWHM_x=4., FWHM_y=4.,theta=0.):
    '''Fit a 2D Gaussian to the data and return amplitude, Gaussian mean, stddev with an option of cutout.'''
    len_y, len_x = data.shape
    range_x = range(len_x)
    range_y = range(len_y)
    x, y = np.meshgrid(range_x, range_y)
    #x0 = len_x / 2 
    #y0 = len_y / 2
    y0,x0 = np.where(data==np.max(data))
    ginit = Gaussian2D(amplitude=1., x_mean=x0, \
        y_mean=y0, x_stddev=FWHM_x*gaussian_fwhm_to_sigma, \
        y_stddev=FWHM_y*gaussian_fwhm_to_sigma, theta=theta)

    fitter = fitting.LevMarLSQFitter()
    results = fitter(ginit, x, y, data, maxiter=1000, acc=1e-10)
    return results

def fit_PRF(data):
    len_y, len_x = data.shape
    range_x = range(len_x)
    range_y = range(len_y)
    x, y = np.meshgrid(range_x, range_y)
    x0 = (len_x / 2)
    y0 = (len_y / 2)

    ginit=IntegratedGaussianPRF(3., x0, y0, 1.)
    fitter = fitting.LevMarLSQFitter()
    results = fitter(ginit, x, y, data, maxiter=1000, acc=1e-08)
    return results


def bootstrap_err(array,iter,average='Median',strn=False):
    '''Get bootstrap errors on a measurement with an option of printing results for papers.'''
    arr=[];confidence=0.68
    for i in range(iter):
        resample = array[np.random.randint(0, len(array), size=len(array))]
        if average=='Mean':    
            arr.append(np.mean(resample))
        else:
            arr.append(np.median(resample))
    results = np.percentile(arr,[25, 50, 75])
    return np.round(results,3)
    #if strn==True:
    #    return print (average+'='+str('%.2f' % (results[1]))+'$_{-'+str('%.3f' % (results[1]-results[0]))+'}^{+'+str('%.3f' % (results[2]-results[1]))+'}$')
    #else:
    #    return np.round(results,3)

def get_beam (radio_map):
    '''Calculate beam size of a radio map.'''
    image=get_map(radio_map)
    bmaj=image[0].header['BMAJ']
    bmin=image[0].header['BMIN']
    w = wcs.WCS(image[0].header)
    # find the pixel size in arcsec
    pix_size = 3600.0*ppps(w)[0] # per pixel in arcsecsonds
    #print ("Image pixel size is ",pix_size, "arcsec")
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
    #print ("Beam area in pixels: ",beam_area)
    return beam_area,pix_size
    print (beam_area,pix_size)

def calc_app_correction(aperture,radiomap):
    image=get_map(radiomap)
    bmaj = image[0].header['BMAJ']*3600.
    pix_size = get_beam(radiomap)[1]
    bmaj=bmaj/pix_size       # pixels
    sigma=bmaj/2.355         # sigma=FWHM/2.355
    aper_p = aperture/pix_size # pixels; aperture is in arcsec
    x_ra=50.1;x_dec=50.66
    data=gaussian_simple(100,100,x_ra,x_dec,sigma,sigma,0)
    #print (data.sum())
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
    #print ("Aperture correction: ",flux)
    return flux


def aper_flux(ra,dec,data,aper_p,wcs_im,beam):
    """ aper_p in pixels """
    re_coord = [ra,dec]
    im_coord = wcs_im.wcs_world2pix([re_coord],0)
    x_ra = im_coord[0][0]
    x_dec = im_coord[0][1]
    ysize,xsize=data.shape
    if x_ra<aper_p or x_dec<aper_p or x_ra>(xsize-aper_p) or x_dec>(ysize-aper_p):
        flux=noise=np.nan
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
        flux=np.sum(subim[mask])/beam
    return flux


def point_rms(ra,dec,data,wcs):
    re_coord = [ra,dec]
    im_coord = wcs.wcs_world2pix([re_coord],0)
    x_ra = int(round(im_coord[0][0]))
    x_dec = int(round(im_coord[0][1]))
    noise = data[x_dec,x_ra]
    return noise

def point_flux(ra,dec,data,aper_p,wcs):
    """ aper is in pixels """
    re_coord = [ra,dec]
    im_coord = wcs.wcs_world2pix([re_coord],0)
    x_ra = im_coord[0][0]
    x_dec = im_coord[0][1]
    ysize,xsize=data.shape
    if x_ra<aper_p or x_dec<aper_p or x_ra>(xsize-aper_p) or x_dec>(ysize-aper_p):
        pflux=np.nan
    else:
        xmin=int(x_ra-(aper_p+1))
        xmax=int(x_ra+(aper_p+1))
        ymin=int(x_dec-(aper_p+1))
        ymax=int(x_dec+(aper_p+1))
        x_pra=x_ra-xmin
        x_pdec=x_dec-ymin
        subim=data[ymin:ymax,xmin:xmax]
        pflux = subim[int(round(x_pdec)),int(round(x_pra))]
    return pflux

def ngp_aper_flux(ra,dec,aper):
    image = "/Users/gur031/hatlas-new/hatlas.mosaic.fits"
    appcorr=calc_app_correction(aper,image)
    flux_map=get_map(image);fmap=flatten(flux_map);wcs_im = wcs.WCS(fmap[0])
    position=SkyCoord(ra, dec, unit="deg")
    cutout_img = Cutout2D(fmap[1], position, (9, 9), wcs=wcs_im)
    subim=cutout_img.data
    beam = get_beam(image)[0]                 # beam area in arcsec
    pixel_size = get_beam(image)[1]
    aper=aper/pixel_size
    H, W = subim.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x_ra=cutout_img.position_cutout[0];x_dec=cutout_img.position_cutout[1]
    mask = (x - x_ra)**2 + (y - x_dec)**2 <= aper**2
    flux_ap=np.sum(subim[mask]/beam)/appcorr
    return flux_ap


def separation(c_ra,c_dec,ra,dec):
    # all values in degrees
    return np.sqrt((np.cos(c_dec*np.pi/180.0)*(ra-c_ra))**2.0+(dec-c_dec)**2.0)


def select_isolated_sources(t,radius):
    t['NN_dist']=np.nan
    for r in t:
        dist=3600.0*separation(r['RA'],r['DEC'],t['RA'],t['DEC'])
        #dist=np.sqrt((np.cos(c_dec*np.pi/180.0)*(t['RA']-r['RA']))**2.0+(t['DEC']-r['DEC'])**2.0)*3600.0
        dist.sort()
        r['NN_dist']=dist[1]

    t=t[t['NN_dist']>radius]
    return t


def get_peak_rms(ra,dec,data,rmsdata,aper_p,wcs_im,beam):
    re_coord = [ra,dec]
    im_coord = wcs_im.wcs_world2pix([re_coord],0)
    x_ra = im_coord[0][0]
    x_dec = im_coord[0][1]
    ysize,xsize=data.shape
    if x_ra<aper_p or x_dec<aper_p or x_ra>(xsize-aper_p) or x_dec>(ysize-aper_p):
        flux_peak=noise=np.nan
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
        flux_peak = np.max(subim)
        subim2=rmsdata[ymin:ymax,xmin:xmax]
        if math.isnan(flux_peak) == True:
            noise =np.nan
        else:
            noise = subim2[np.where(subim==flux_peak)][0]
    return flux_peak,noise
def match_peaks(data, model, cval=0):
    '''Shift the data so that the index of the maximum value in the data matches up with that of the model. '''
    '''This is needed to ensure accurate subtraction.'''
    data_max = data.argmax()
    model_max = model.argmax()
    data_shape = data.shape
    model_shape = model.shape
    data_peak_row, data_peak_column = np.unravel_index(data_max, data_shape)
    model_peak_row, model_peak_column = np.unravel_index(model_max, model_shape)
    shifting = (model_peak_row - data_peak_row, model_peak_column - data_peak_column)
    shifted_data = shift(data, shift=shifting, cval=cval)
    numbers = {'d': (data_peak_row, data_peak_column), 's': shifting, 'm': (model_peak_row, model_peak_column)}
    print('The data were shifted from {d} by {s} to {m}.'.format(**numbers))
    return shifted_data

def regrid(data, new_size=5, normalise=True):
    '''Map the data onto a larger array.'''
    l, w = data.shape
    new_l = l * new_size
    new_w = w * new_size
    new_dimensions = []
    for old_shape, new_shape in zip(data.shape, (new_l, new_w)):
        new_dimensions.append(np.linspace(0, old_shape - 1, new_shape))
    coordinates = np.meshgrid(*new_dimensions, indexing='ij')
    new_data = map_coordinates(data, coordinates)
    #new_data = new_data / np.max(new_data) if normalise else new_data
    return new_data  # this function was tested and worked as expected

def make_model(data, amplitude=1, sigma_x=30, sigma_y=30, theta=0, offset=0):
    '''Fit a model to the data.'''
    len_x, len_y = data.shape
    range_x = range(len_x)
    range_y = range(len_y)
    x, y = np.meshgrid(range_x, range_y)
    x0 = len_x / 2
    y0 = len_y / 2
    data_ravel = data.ravel()
    p0 = (amplitude, x0, y0, sigma_x, sigma_y, theta, offset)
    popt, pcov = opt.curve_fit(gaussian, (x, y), data_ravel, p0=p0)
    model = gaussian((x, y), *popt)
    model_reshape = model.reshape(len_x, len_y)
    return model

'''
def projected_beam(bmin,bmaj,pos_angle,pos_angle_ref_phase_centre):
    proj_beam = (bmin*bmaj)/np.sqrt((bmaj*np.sin(pos_angle-pos_angle_ref_phase_centre))**2+(bmin*np.cos(pos_angle-pos_angle_ref_phase_centre))**2)
    return proj_beam

def beam_smearing(peak_flux,central_freq,bandwidth,proj_beam,beam_centre_distance):
    central_freq = 8.874907407407E+08
    bandwidth = 1e+6


import numpy as np
import numexpr as ne

def gaussian(xsize,ysize,x0,y0,sx,sy,pa):
    X, Y = np.meshgrid(np.arange(0,xsize,1.0), np.arange(0,ysize,1.0))
    pa=-(90.0+pa) # N through E
    pa*=np.pi/180.0
    a=0.5*((np.cos(pa)/sx)**2.0+(np.sin(pa)/sy)**2.0)
    b=0.25*((-np.sin(2*pa)/sx**2.0)+(np.sin(2*pa)/sy**2.0))
    c=0.5*((np.sin(pa)/sx)**2.0+(np.cos(pa)/sy)**2.0)

    return ne.evaluate('exp(-(a*(X-x0)**2.0+2*b*(X-x0)*(Y-y0)+c*(Y-y0)**2.0))')/(2*np.pi*sx
*sy)

bmaj=10.0/1.5 # pixels
sigma=bmaj/2.355
aper_r = 10.00 #arcsec
aper_p = aper_r/1.5

x_ra=50.1
x_dec=50.66

data=gaussian(100,100,x_ra,x_dec,sigma,sigma,0)

print data.sum()

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

print flux

'''


