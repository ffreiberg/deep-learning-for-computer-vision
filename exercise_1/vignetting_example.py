#!/usr/bin/python

# Example code to demonstrate vignetting. Of course, parameters are different
# to the training/test images provided with the exercise.

import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import code


# function for vignetting, hard-coded parameters (I'm lazy)
def vignetting( I ):

    a0 = 1.0
    a1 = -0.3
    a2 = 0.05
    a3 = -0.25
    a4 = -0.4
    a5 = -0.05
    a6 = 0.1
    
    # compute vignetting factor on complete grid
    W = I.shape[1]
    H = I.shape[0]
    wc = W/2
    hc = H/2
    
    xv,yv = np.meshgrid( np.arange( W ) - wc, np.arange( H ) - hc)

    r = np.sqrt( xv ** 2 + yv ** 2 ) / np.sqrt( wc**2 + hc**2 )
    s = a0 + a1*r + a2* (r **2 ) + a3 * (r ** 3) + a4 * (r ** 4) + a5 * (r**5) + a6 * (r**6) # and so on
    
    #code.interact( local=locals() )
    
    J = np.zeros( I.shape, np.float32 )
    J[ :,:,0 ] = s * I[ :,:,0 ]
    J[ :,:,1 ] = s * I[ :,:,1 ]
    J[ :,:,2 ] = s * I[ :,:,2 ]
    return J


    

# BEGIN MAIN PROGRAM
img = imread('cat_01.jpg')
img_vignetted = vignetting( img )

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_vignetted))
plt.show()
