# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:49:01 2017

@author: Dustin
"""
import numpy as np
import requests
from PIL import Image
import os
from io import BytesIO
from numpy import random
from scipy import ndimage
import timeit
import json
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import zoom
import cv2
# import imagehash as ih

base_url = "https://api.scryfall.com"

def GrabCardData(set_name, card_number):
    card_url = base_url + "/cards/" + set_name + '/' + str(card_number)
    response = requests.get(card_url)

    return json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))

def CardFileName( card_name ):
    return '/hdd/Programming/magic-image-classification/images/' + card_name + '.png'

def DirtyCardFileName( card_name ):
    return '/home/adam/Pictures/MtgDataset/2XM/' + card_name + "/" + card_name

def GetCurrentCardDirectory(card_name):
    return '/home/adam/Pictures/MtgDataset/2XM/' + card_name + "/"

def GetNumFilesInDir(dir):
    files = os.listdir(dir)
    return len(files)

def CardName( set_name, card_number ):
    cname = set_name + '_' + str(card_number)
    return cname

def CallImage( set_name, card_number, lut ):
    # Call the Scryfall API for a png image if it is not currently saved in the
    #       image folder
    current_card = lut[card_number]

    if os.path.isfile(CardFileName(current_card)):
        return current_card
    else:
        current_card = GrabCardData(set_name, card_number + 1) # add 1 to the card number as the array/lut starts at 0 
        response = requests.get(current_card.image_uris.normal)
        im = Image.open(BytesIO(response.content))
        
        im.save(CardFileName(current_card.name))
        print( "Called API")

        return current_card.name

def RemoveImage( card_name ):
    # Delete an image from the image folder
    if os.path.isfile(CardFileName(card_name)):
        os.remove( CardFileName(card_name) )

def PullImage( card_name ):
    # Pull an image from the images folder and give it as a np.array
    im = Image.open(CardFileName(card_name))
    im = np.array(im)[:,:,0:3]
    return im

def s_PullImage( card_name ):
    # Pull an image from the images folder and give it as a np.array
    t0 = timeit.default_timer()
    im = Image.open(CardFileName(card_name))
    topen = timeit.default_timer(); sopen = topen - t0
    im = np.array(im)[:,:,0:3]
    tarra = timeit.default_timer(); sarra = tarra - topen
    times = np.array( (sopen, sarra) )
    return [im,times]

def s_plot(img):
    Image.fromarray(img).show()

def line( x0, y0, x1, y1):
    x = np.arange( x0, x1+1 , 1)
    m = (y1 - y0)/(x1 - x0)
    y = ( m*( x - x0) + y0 ).astype('int')
    return x,y

def addline( im ):
    # Add a scratch mark to a np.array image.
    thick = int( np.random.uniform( 1, im.shape[1]/100) )
    color = random.randint(2,size = 1)[0]*np.array((255,255,255))
    x0 = int(np.random.uniform( 0, im.shape[0]-1 ))
    y0 = int(np.random.uniform( 0, im.shape[1] ))
    x1 = int(np.random.uniform( x0+1, im.shape[0] ))
    y1 = int(np.random.uniform( 0, im.shape[1] ))
    x = np.array([])
    y = np.array([])
    for i in range(thick):
        xn,yn = line( x0, (y0+i)%im.shape[1] ,x1, (y1 + i)%im.shape[1])
        x = np.append( x, xn ).astype('int')
        y = np.append( y, yn ).astype('int')
    im1 = np.copy(im[:,:,:])
    im1[x,y,:] = color
    return(im1)

def addcircle( im ):
    # Adds a smudge to a np.array image
    r = int( np.random.uniform( 1, im.shape[1]/50))
    color = random.randint(2,size = 1)[0]*np.array((255,255,255))
    a = int(np.random.uniform( 0, im.shape[0] ))
    b = int(np.random.uniform( 0, im.shape[1] ))
    nx,ny = im[:,:,1].shape
    x,y = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= r*r
    im1 = np.copy(im[:,:,:])
    im1[mask,:] = color
    return(im1)

def addsaltpepper(im):
    # Adds a layer of salt and pepper noise to a np.array
    prob = np.random.uniform( 0, 0.01)
    rnd = np.random.rand(im.shape[0], im.shape[1])
    im1 = im.copy()
    im1[rnd < prob] = 0
    im1[rnd > 1 - prob] = 255
    return im1

def DirtyImage( im ):
    # An assembly of filters and objects to add to an np.array image to 
    #       randomly produce a new, dirtier image. Outputs as a np.array
    im1 = np.copy(im[:,:,:])
    
    for i in range( int( random.uniform(0, 20) ) ):
        im1 = addline(im1)
    for i in range( int( random.uniform(0, 20) ) ):
        im1 = addcircle(im1)
    im1 = addsaltpepper(im1)
    #im1 = ndimage.filters.median_filter( im1, int(random.exponential(3)+1))
    im1 = ndimage.filters.gaussian_filter( im1, random.exponential(1))
    #im1 = ndimage.rotate( im1, random.uniform(-2.5, 2.5) )
    im1 = np.array(Image.fromarray(im1).rotate(random.uniform(-10,10)))
    im1 = cv2_clipped_zoom(im1, random.uniform(0.75, 1.0))
    return im1

def s_DirtyImage( im ):
    t0 = timeit.default_timer()
    im1 = np.copy(im[:,:,:])
    tcopy = timeit.default_timer(); scopy = tcopy - t0
    for i in range( int( random.uniform(0, 20) ) ):
        im1 = addline(im1)
    tline = timeit.default_timer(); sline = tline - tcopy
    for i in range( int( random.uniform(0, 20) ) ):
        im1 = addcircle(im1)
    tcirc = timeit.default_timer(); scirc = tcirc - tline
    im1 = addsaltpepper(im1)
    tsalt = timeit.default_timer(); ssalt = tsalt - tcirc
    #im1 = ndimage.filters.median_filter( im1, int(random.exponential(3)+1))
    im1 = ndimage.filters.gaussian_filter( im1, random.exponential(1))
    tgaus = timeit.default_timer(); sgaus = tgaus - tsalt
    #im1 = ndimage.rotate( im1, random.uniform(-2.5, 2.5) )
    im1 = np.array(Image.fromarray(im1).rotate(random.uniform(-2.5,2.5)))
    trota = timeit.default_timer(); srota = trota - tgaus
    times = np.array( (scopy, sline, scirc, ssalt, sgaus, srota) )
    return [im1, times]

# def HashImage( im ):
#     # Convert a numpy array image into a hash, convert the hexidecimal into integers
#     ph = ih.phash(Image.fromarray(im[:,:,:]))
#     vint = np.vectorize(int)
#     iph = vint(np.array(list(str(ph))).astype(str),16)
#     return iph


def d_reshape( im, width = 84, length = 117):
    # Reshape the np.array image into a pre-determined size and into a 1D array
    im1 = Image.fromarray(im)
    rm1 = im1.resize( (width, length), Image.ANTIALIAS )
    arm1 = np.array(rm1)
    arm1 = arm1[None,:,:,:]
    #fm1 = np.reshape(arm1, 117*84*3)
    return arm1

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result