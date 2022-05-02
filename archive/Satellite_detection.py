# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:43:10 2021

This program uses astropy routines to identify satellites in ROO images; similar to photometry pipeline, but not relying on SCAMP and Sextractor (that need linux)

The plan is twofold;
    1. serve as a testing program against any ML routines that Sai develops to find satellites
    2. serve as an operational program that points the telescope back at satellites after it's found them. This will hopefully optimize the GEOsearchgrid routine that currently 
    takes way too long to scan... because it's taking 5 images of nothing, a lot of the time
    
In terms of inputs, this code needs the "fits_folder" that contains the FITS that you want processed, and empty "saved_pngs_dir" and "saved_csvs" to store the outputs.

You also need to set the maximum eccentricity for the ellipses around each source identified, and a minimum orientation difference between stars and potential satellites.

Whether the program counts a possible satellite or not depends on either of these options being fulfilled.

To further adjust the success rates, you can also change the values for "npixels" and "threshold" that are used to detect sources.


@author: Dr Brett Carter, RMIT University
"""
from astropy.io import fits
import sys
import numpy as np
from astropy import units as u


import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)


import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.background import Background2D, MedianBackground
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.segmentation import detect_sources,deblend_sources
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.segmentation import SourceCatalog
import os
import csv

fits_folder = "FITS/"
saved_pngs_dir = "PNGs/"
saved_csvs = "CSVs/"

#maximum eccentricity for ellipse around source to be considered a possible satellite
max_ecc = 0.8
max_ecc = 0.75

#minimum orientation difference (from the mean across the image) in order to be considered a possible satellite
orient_diff_min = 10 * u.deg

#minimum number of sources in an image to consider
min_sources = 10

#counter for the number of satellites found
count = 0
file_list = []
n_sats_file = []

for image_file in os.listdir(fits_folder):
    
    if image_file.endswith("fit"):
        print("Processing " +image_file)
    
        #fits.info(image_file)

        data1 = fits.getdata(fits_folder+image_file)
        data = np.float64(data1)
        
        #for the issue of sources being identified on the edges of the images
        dims = data.shape
        dpix = 20   #width of mask on each axis in # of pixels
        #x-axis mask
        xmin_window = 0 + dpix
        xmax_window = dims[1] - dpix
        #y-axis mask
        ymin_window = 0 + dpix
        ymax_window = dims[0] - dpix
        
        #sys.exit()

        #----------------------------------------------
        #this threshold technique appears to be rather slow...
        #from photutils.segmentation import detect_threshold
        #threshold = detect_threshold(data, nsigma=2.)

    
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
        data -= bkg.background  # subtract the background
        threshold = 3. * bkg.background_rms  # above the background

    
        sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        kernel.normalize()
        npixels = 25
        
        segm = detect_sources(data, threshold, npixels=25, kernel=kernel)
        
        if segm is None:
            print("No sources found... bad image")
            
        if segm is not None:
            print("Deplending now.")
            segm_deblend = deblend_sources(data, segm, npixels=npixels,
                                           kernel=kernel, nlevels=32,
                                           contrast=0.001)
                
            print("Deplending complete.")
                
                
            norm = ImageNormalize(stretch=SqrtStretch())
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(35, 25))
            ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm);
            ax1.set_title('Data');
            cmap = segm.make_cmap(seed=123)
            ax2.imshow(segm, origin='lower', cmap=cmap, interpolation='nearest');
            ax2.set_title('Segmentation Image');
                
                
            cat = SourceCatalog(data, segm_deblend)
                
            tbl = cat.to_table()
                
            #writing table to csv file for further analysis...
            tbl.write(saved_csvs+image_file+".csv",format = "ascii.csv",overwrite=True)
                
            #cat_size = sys.getsizeof(cat)
                
            sats_in_image = 0.
            
            #mean_orient = np.mean(cat.orientation)
            #trying median, less sensitive to outliers... (except where there are few sources..., restricting to no less than 10 sources)
            if len(cat.orientation) > min_sources:
                mean_orient = np.median(cat.orientation)
            
                for i, element in enumerate(cat):
                    aperture = cat.kron_aperture[i]
                    ecc = cat.eccentricity[i]
                    orient_diff = abs(cat.orientation[i]-mean_orient)
                    
                    
                    #I had to insert this line because some of the apertures were being spat out as "Nonetype"... I'm assuming because the image quality??? But, images cleaned using AstroImageJ didn't work...
                    #if aperture is not None and (ecc < max_ecc or orient_diff > orient_diff_min):
                    if aperture is not None and (ecc < max_ecc or orient_diff > orient_diff_min) and (aperture.positions[0] > xmin_window and aperture.positions[0] < xmax_window) and (aperture.positions[1] > ymin_window and aperture.positions[1] < ymax_window):
                    
                        #if aperture is not None:
                            #aperture.plot(axes=ax1, color='white', lw=1.5)
                            aperture.plot(axes=ax2, color='white', lw=1.5)
                            
                            xpos = aperture.positions[0]
                            ypos = aperture.positions[1]
                            
                            ax2.plot(xpos,ypos,color = 'red', marker = 'o', markersize = 22, mfc='none');
                            print("Possible satellite found")
                            count += 1
                    
                            sats_in_image += 1
                    
                    
            plt.ioff()
            fig.savefig(saved_pngs_dir+image_file+".png");
            plt.close("all")
                            
            data = []
            data1 = []
            segm = []
            segm_deblend = []
            
            file_list.append([image_file,str(sats_in_image)])
                    
print("Number of possible satellites found in batch: ", count)
file_list.append(["Total",str(count)])


with open(saved_csvs+"Overview.csv", 'w', newline='') as f:
     write = csv.writer(f)
     for val in file_list:
         write.writerow(val)

print("Results saved in "+saved_csvs+"Overview.csv")
