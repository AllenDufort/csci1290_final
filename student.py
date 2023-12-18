'''
Project: Texture - student.py
CSCI 1290 Computational Photography, Brown U.
Written by Emanuel Zgraggen.
Converted to Python by Trevor Houchens.

'''

import numpy as np
import cv2
import math
import random
from skimage.color import rgb2gray

# ------------------------TEXTURE FUNCTIONS------------------------

def image_texture(source, texture, outsize, tilesize, overlapsize, n_iter, quiet, mask):
    '''
    Outputs an image that looks like the source but is created from samples
    from the texture image
    '''

    adjsize = tilesize - overlapsize
    imout = np.zeros((math.ceil(outsize[0] / adjsize) * adjsize + overlapsize, math.ceil(outsize[1] / adjsize) * adjsize + overlapsize, source.shape[2]))

    # if no mask is given, entire image is rendered as painting
    if mask.shape == None:
        mask = np.ones((imout.shape[0], imout.shape[1], 1), dtype=bool)
     
    for n in range(n_iter):
        # decrease tilesize for later runs
        if n > 0:
            tilesize = math.ceil(tilesize * 0.7)
            overlapsize = math.ceil(tilesize / 6)
            adjsize = tilesize - overlapsize

        # Gradually increase alpha from 0.1 to 0.9 throughout the iterations
        if n_iter > 1:
            alpha = 0.8 * ((n) / (n_iter-1)) + 0.1
        # catch div by zero if there is only one iteration
        else:
            alpha = 0.1

        imout_mask = np.zeros((imout.shape[0], imout.shape[1]), dtype=bool)
        
        # We made the output image slightly larger than the source 
        # so now we'll make the source the same size by padding

        source = np.pad(source, [(0,imout.shape[0]-source.shape[0]),(0,imout.shape[1]-source.shape[1]), (0,0)], mode='symmetric')
    

        # iterate over top left corner indices of tiles
        for y in range(0,imout.shape[0]-tilesize+1, adjsize):
            if np.count_nonzero(mask[y:y+tilesize, :, 0]) == 0:
                imout_mask[y:y+tilesize,:] = True
                continue
            else:
                for x in range(0,imout.shape[1]-tilesize+1, adjsize):
                    if np.count_nonzero(mask[y:y+tilesize, x:x + tilesize, 0]) == 0:
                        imout_mask[y:y+tilesize,x:x+tilesize] = True
                        continue
                    else:
                        # patch we want to fill
                        to_fill = imout[y:y+tilesize, x:x + tilesize]

                        # mask of what part has been filled
                        to_fill_mask = imout_mask[y:y+tilesize, x:x+tilesize]

                        # get the patch we want to insert
                        patch_to_insert = get_patch_to_insert_transfer(tilesize, overlapsize, to_fill, to_fill_mask, texture, alpha, source[y:y + tilesize, x:x+tilesize])

                        # update the image and the mask
                        imout[y:y+tilesize, x:x+tilesize] = patch_to_insert
                        imout_mask[y:y+tilesize,x:x+tilesize] = True

                    if not quiet:
                        cv2.imshow("Output Image", imout)
                        cv2.waitKey(1)
        
        if not quiet:
            cv2.waitKey(1000)
    
    if not quiet:
        cv2.waitKey(5000)

    imout = imout[:outsize[0],:outsize[1]]

    return imout


def get_patch_to_insert_transfer(tilesize, overlapsize, to_fill, to_fill_mask, texture, alpha, source):
    '''
    Returns patch to insert for texture transfer. This patch is chosen based on how well it matches the
    existing texture, and its correspondence to the source image.

    Arguments:
    tilesize     - the size of the tile
    overlapsize  - the overlap between tiles
    to_fill      - the section of the output image that we want to fill
    to_fill_mask - a mask of which part of to_fill has already been filled in previous steps
    texture      - a sample of the texture that we are using to replicate the source
    source       - the section of the source that corresponds to the to_fill patch

    Output:
    A patch of texture insert into the output image
    '''
    
    # TODO: Implement this function
    # We can use the same set of patch selection methods as before, but we also want our synthesized texture
    # to have the same intensity as the source image. What selection criteria or cost might we use?

     # Calculate dimensions of the texture and the to_fill mask
    width = texture.shape[1]
    height = texture.shape[0]

    width_mask = to_fill_mask.shape[1]
    height_mask = to_fill_mask.shape[0]

    # Calculate the maximum possible integer values for width and height
    max_width_integer = np.floor(width/tilesize)
    max_height_integer = np.floor(height/tilesize)

    width_range = range(int(max_width_integer))
    height_range = range(int(max_height_integer))

    # Convert source to grayscale for intensity matching
    source_gray = rgb2gray(source)

    patch_indices = []
    errors = []

    # Loop through possible patch locations
    for i in height_range:
        for j in width_range:
            patch_indices.append(tuple((i, j)))
            width_range_start = j*tilesize
            height_range_start = i*tilesize
            patch = texture[height_range_start:height_range_start+tilesize, width_range_start:width_range_start+tilesize, :]

            patch_gray = rgb2gray(patch)

            #Overlap error - same as get ssd patch - only accounts for overlap because to_fill_mask is mostly 0's
            to_fill_mask_extend = np.repeat(to_fill_mask[:, :, np.newaxis], 3, axis=2)
            patch_overlap = np.multiply(patch, to_fill_mask_extend)
            diff = patch_overlap - to_fill
            squared = np.square(diff)
            reg_error = np.sum(squared)

            #To fill though contains the image from the last iteration
            reverse_mask = np.abs(1- to_fill_mask)
            reverse_mask_extend = np.repeat(reverse_mask[:, :, np.newaxis], 3, axis=2)
            patch_previous = np.multiply(patch, reverse_mask_extend)
            diff2 = patch_previous - to_fill
            squared2 = np.square(diff2)
            previous_error = np.sum(squared2)

            #Correspondence error
            diff3 = patch_gray - source_gray
            squared3 = np.square(diff3)
            corr_error = np.sum(squared3)

            # Combine errors with the trade-off parameter alpha
            error1 = alpha * (reg_error + previous_error)
            error2 = ( 1 - alpha) * corr_error

            error = error1 + error2

            errors.append(error)
    
    # Select the N patches with the lowest errors
    errors, patch_indices = zip(*sorted(zip(errors, patch_indices)))
    N = 3
    errors_N = errors[0:N]
    patch_indices_N = patch_indices[0:N]

    # Randomly select one patch from the N patches
    select = np.random.randint(N)
    index_select = patch_indices_N[select]

    width_index = index_select[1]
    height_index = index_select[0]

    width_range_start = width_index*tilesize
    height_range_start = height_index*tilesize

    # Extract the selected patch
    patch = texture[height_range_start:height_range_start+tilesize, width_range_start:width_range_start+tilesize, :]

    # Handle special cases for blending based on overlap in the mask
    if np.sum(to_fill) == 0:
        return patch
    elif (to_fill_mask[0, width_mask-1] == 0 and to_fill_mask[height_mask-1, 0] == 1): 
        # Just find cut to the left 

        to_fill_select = to_fill[:, 0:overlapsize, :]
        patch_select = patch[:, 0:overlapsize, :]
        
        diff = np.abs(to_fill_select - patch_select)

        E_matrix = np.sum(diff, axis=2)

        M_matrix = E_matrix.copy()

        for y in range(1,M_matrix.shape[0]):
            for x in range(M_matrix.shape[1]):
                if x == 0:
                    M_matrix[y, x] = M_matrix[y, x] + min([M_matrix[y-1, x], M_matrix[y-1, x+1]])
                elif x == M_matrix.shape[1]-1:
                    M_matrix[y, x] = M_matrix[y, x] + min([M_matrix[y-1, x], M_matrix[y-1, x-1]])
                else:
                    M_matrix[y, x] = M_matrix[y, x] + min([M_matrix[y-1, x], M_matrix[y-1, x+1], M_matrix[y-1, x-1]])

        final_mask = np.ones(patch.shape)

        y = M_matrix.shape[0] - 1

        row = M_matrix[y, :]
        index = np.argmin(row)
        final_mask[y, 0:index, :] = 0

        while y >= 1:

            if index == 0:
                select_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1

            elif index == M_matrix.shape[1]-1:
                select_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index-1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index-1

                
            else:
                select_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1], M_matrix[y-1, index-1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1
                elif select_index == 2:
                    next_index = index - 1
                
                

            final_mask[y-1, 0:next_index, :] = 0
            index = next_index
            y = y - 1

        return np.multiply(final_mask, patch) + np.multiply(1-final_mask, to_fill)
    

    elif (to_fill_mask[0, width_mask-1] == 1 and to_fill_mask[height_mask-1, 0] == 0): 
        # Just find cut above

        to_fill_select = to_fill[0:overlapsize, :, :]
        patch_select = patch[0:overlapsize, :, :]
        
        diff = np.abs(to_fill_select - patch_select)

        E_matrix = np.sum(diff, axis=2)

        M_matrix = E_matrix.copy()

        for y in reversed(range(0,M_matrix.shape[1]-2)):
            for x in range(M_matrix.shape[0]):
                if x == 0:
                    M_matrix[x, y] = M_matrix[x, y] + min([M_matrix[x, y+1], M_matrix[x+1, y+1]])
                elif x == M_matrix.shape[0]-1:
                    M_matrix[x, y] = M_matrix[x, y] + min([M_matrix[x, y+1], M_matrix[x-1, y+1]])
                else:
                    M_matrix[x, y] = M_matrix[x, y] + min([M_matrix[x, y+1], M_matrix[x+1, y+1], M_matrix[x-1, y+1]])

        final_mask = np.ones(patch.shape)


        y = 0 

        column = M_matrix[:, y]
        index = np.argmin(column)

        final_mask[0:index, y, :] = 0

        while y <= M_matrix.shape[1] - 2:

            if index == 0:
                select_index = np.argmin([M_matrix[index, y+1], M_matrix[index+1, y+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1

                
            elif index == M_matrix.shape[0]-1:
                select_index = np.argmin([M_matrix[index, y+1], M_matrix[index-1, y+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index-1

              
            else:
                select_index = np.argmin([M_matrix[index, y+1], M_matrix[index+1, y+1], M_matrix[index-1, y+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1
                elif select_index == 2:
                    next_index = index - 1
                
                

            final_mask[0:next_index, y+1, :] = 0
            index = next_index
            y = y + 1

        return np.multiply(final_mask, patch) + np.multiply(1-final_mask, to_fill)
    
    elif (to_fill_mask[0, width_mask-1] == 1 and to_fill_mask[height_mask-1, 0] == 1):
        # Find both left and above cuts
        
        to_fill_select = to_fill[0:overlapsize, :, :]
        patch_select = patch[0:overlapsize, :, :]
        
        diff = np.abs(to_fill_select - patch_select)

        E_matrix = np.sum(diff, axis=2)

        M_matrix = E_matrix.copy()

        for y in reversed(range(0,M_matrix.shape[1]-2)):
            for x in range(M_matrix.shape[0]):
                if x == 0:
                    M_matrix[x, y] = M_matrix[x, y] + min([M_matrix[x, y+1], M_matrix[x+1, y+1]])
                elif x == M_matrix.shape[0]-1:
                    M_matrix[x, y] = M_matrix[x, y] + min([M_matrix[x, y+1], M_matrix[x-1, y+1]])
                else:
                    M_matrix[x, y] = M_matrix[x, y] + min([M_matrix[x, y+1], M_matrix[x+1, y+1], M_matrix[x-1, y+1]])

        final_mask = np.ones(patch.shape)

        
        y = 0 

        column = M_matrix[:, y]
        index = np.argmin(column)
        final_mask[0:index, y, :] = 0

        while y <= M_matrix.shape[1] - 2:

            if index == 0:
                select_index = np.argmin([M_matrix[index, y+1], M_matrix[index+1, y+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1

                
            elif index == M_matrix.shape[0]-1:
                select_index = np.argmin([M_matrix[index, y+1], M_matrix[index-1, y+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index-1

               
            else:
                select_index = np.argmin([M_matrix[index, y+1], M_matrix[index+1, y+1], M_matrix[index-1, y+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1
                elif select_index == 2:
                    next_index = index - 1
                
                

            final_mask[0:next_index, y+1, :] = 0
            index = next_index
            y = y + 1

        patch = np.multiply(final_mask, patch) + np.multiply(1-final_mask, to_fill)

        to_fill_select = to_fill[:, 0:overlapsize, :]
        patch_select = patch[:, 0:overlapsize, :]
        
        diff = np.abs(to_fill_select - patch_select)

        E_matrix = np.sum(diff, axis=2)

        M_matrix = E_matrix.copy()

        for y in range(1,M_matrix.shape[0]):
            for x in range(M_matrix.shape[1]):
                if x == 0:
                    M_matrix[y, x] = M_matrix[y, x] + min([M_matrix[y-1, x], M_matrix[y-1, x+1]])
                elif x == M_matrix.shape[1]-1:
                    M_matrix[y, x] = M_matrix[y, x] + min([M_matrix[y-1, x], M_matrix[y-1, x-1]])
                else:
                    M_matrix[y, x] = M_matrix[y, x] + min([M_matrix[y-1, x], M_matrix[y-1, x+1], M_matrix[y-1, x-1]])

        final_mask = np.ones(patch.shape)

        y = M_matrix.shape[0] - 1

        row = M_matrix[y, :]
        index = np.argmin(row)
        final_mask[y, 0:index, :] = 0

        while y >= 1:

            if index == 0:
                select_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1

                
            elif index == M_matrix.shape[1]-1:
                select_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index-1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index-1

                
            else:
                select_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1], M_matrix[y-1, index-1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1
                elif select_index == 2:
                    next_index = index - 1
                
                

            final_mask[y-1, 0:next_index, :] = 0
            index = next_index
            y = y - 1

        return np.multiply(final_mask, patch) + np.multiply(1-final_mask, to_fill)