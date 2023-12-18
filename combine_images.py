#Inputs:
    # out_put_file_name is a string 
    # all other inputs are arrays (I used cv2.imread to load each of them)


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def combine_images(mask, sky_output, full_image):
    mask = mask.astype(np.float32) / 255.
    sky_output = sky_output.astype(np.float32) / 255.
    full_image = full_image.astype(np.float32) / 255.

    inverse_mask = abs(mask - 1)
    inverse_mask = inverse_mask.astype(np.float32)

    sky = mask * sky_output
    foreground = inverse_mask * full_image 

    cv2.imshow("sky", sky*255)
    cv2.waitKey(1000)
    cv2.imshow("foreground", foreground*255)
    cv2.waitKey(1000)
    final_img = (sky + foreground) * 255

    cv2.imshow("All", final_img)
    cv2.waitKey(2000)

    return final_img 