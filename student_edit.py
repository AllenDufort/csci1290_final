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


# ------------------------SYNTHESIS FUNCTIONS------------------------

def image_synthesis(sample, outsize, tilesize, overlapsize, method, quiet):
    '''
    Takes in an sample image, and outputs an image with dimensions that are specified
    by outsize (plus channels) that has texture synthesized from the sample

    Arguments:
    sample-       sample texure image
    outsize-      size of result image
    tilesize-     size of the tiles to use (patch size)
    overlapsize-  size of the overlap region (same in each direction -> left
                  and above)
    texture-      texure image
    method-       1 = random, 2 = best ssd, 3 = best ssd + minimum error boundary cut
    quiet-        no images shown if true

    Output:
        synthesized image of size outsize[1] x outsize[2] x channels 
    '''

    # The amount of additional space each tile takes up (accounting for overlap)
    adjsize = tilesize - overlapsize

    # imout is the array we will fill. The size is slightly larger than outsize to start
    # since it will be filled with an integer number of tiles
    imout = np.zeros((int(math.ceil(outsize[0] / adjsize) * adjsize + overlapsize), int(math.ceil(outsize[1]/adjsize) * adjsize + overlapsize), sample.shape[2]))
    imout_mask = np.zeros((int(math.ceil(outsize[0] / adjsize) * adjsize + overlapsize), int(math.ceil(outsize[1]/adjsize) * adjsize + overlapsize)), dtype=bool)


    # iterate over each tile
    for y in range(0, outsize[0], adjsize):
        for x in range(0, outsize[1], adjsize):
        
            # the patch of imout that we want to fill - for all tiles except the first this region will have some nonzero values
            to_fill = imout[y:y + tilesize, x:x + tilesize, :]

            # the mask for this patch (includes left and above overlap)
            to_fill_mask = imout_mask[y:y + tilesize, x:x + tilesize]

            # get the patch that we want to insert into imout
            patch_to_insert = get_patch_to_insert_synthesis(method, tilesize, overlapsize, to_fill, to_fill_mask, sample)

            # update result image and mask
            imout[y:y + tilesize, x:x + tilesize,:] = patch_to_insert
            imout_mask[y:y + tilesize, x:x + tilesize] = 1

            # Pause briefly to show the result
            if not quiet:
                cv2.imshow("Output Image", imout)
                cv2.waitKey(10)
    
    # Pause for 5 seconds to show the result
    if not quiet:
        cv2.waitKey(5000)

    imout = imout[:outsize[0],:outsize[1]]
    
    return imout


def get_patch_to_insert_synthesis(method, tilesize, overlapsize, to_fill, to_fill_mask, texture):
    '''
    Finds a patch of texture from a sample image that fits into a provided tile from 
    a partially synthesized texture image.

    Arguments:
    method:       1 = random, 2 = best ssd, 3 = best ssd + minimum error boundary cut
    tilesize:     size of the tiles to use (patch size)
    overlapsize:  size of the overlap region (same in each direction -> left
                   and above)
    to_fill:      patch of the synthesized image at current location
    to_fill_mask: mask of the synthesized image at current patch location 
    sample:       sample texture image
    
    Output:
        patch of size |tilesize|x|tilesize|x|channels| that will be inserted 
        at current location (needs to include to overlap region). 
    '''
    if method == 1:
        patch = get_random_patch(texture, tilesize)

    if method == 2:
        patch = get_ssd_patch(texture, tilesize, overlapsize, to_fill, to_fill_mask)

    if method == 3:
        patch = get_min_cut_patch(texture, tilesize, overlapsize, to_fill, to_fill_mask)

    return patch


def get_random_patch(texture, tilesize):
    '''
    Returns a tile sampled uniformly at random locations from the texture image
    
    Arguments:
    texture  - the texture image
    tilesize - the desired tile size

    Returns-
    A tilesize x tilesize section of texture
    '''
    width = texture.shape[1]
    height = texture.shape[0]

    max_width_integer = np.floor(width/tilesize)
    max_height_integer = np.floor(height/tilesize)

    width_index = np.random.randint(max_width_integer)
    height_index = np.random.randint(max_height_integer)

    width_range_start = width_index*tilesize
    height_range_start = height_index*tilesize

    return texture[height_range_start:height_range_start+tilesize, width_range_start:width_range_start+tilesize, :]


def get_ssd_patch(texture, tilesize, to_fill, to_fill_mask):
    '''
    Returns a patch selected from the texture image that has a low SSD with the
    already filled part of the tile.

    Arguments:
    texture      - the texture image
    tilesize     - the desired tilesize
    overlapsize  - the overlap between tiles
    to_fill      - the tile that is being filled in the synthesized image
    to_fill_mask - a mask for to_fill marking which parts have been filled

    Returns:
    A tilesize x tilesize section of texture
    '''

    # TODO: Implement this function
    # Use to_fill_mask to see which parts of to_fill have already been filled
    # The goal is to minimize the ssd between the chosen patch and the already
    # filled parts of to_fill.
    
    width = texture.shape[1]
    height = texture.shape[0]

    max_width_integer = np.floor(width/tilesize)
    max_height_integer = np.floor(height/tilesize)

    width_range = range(int(max_width_integer))
    height_range = range(int(max_height_integer))

    patch_indices = []
    errors = []

    for i in height_range:
        for j in width_range:
            patch_indices.append(tuple((i, j)))
            width_range_start = j*tilesize
            height_range_start = i*tilesize
            patch = texture[height_range_start:height_range_start+tilesize, width_range_start:width_range_start+tilesize, :]

            to_fill_mask_extend = np.repeat(to_fill_mask[:, :, np.newaxis], 3, axis=2)
            patch_overlap = np.multiply(patch, to_fill_mask_extend)
            diff = patch_overlap - to_fill
            squared = np.square(diff)

            ssd = np.sum(squared)
            errors.append(ssd)
    
    errors, patch_indices = zip(*sorted(zip(errors, patch_indices)))
    N = 3

    errors_N = errors[0:N]
    patch_indices_N = patch_indices[0:N]

    select = np.random.randint(N)
    index_select = patch_indices_N[select]

    width_index = index_select[1]
    height_index = index_select[0]

    width_range_start = width_index*tilesize
    height_range_start = height_index*tilesize

    return texture[height_range_start:height_range_start+tilesize, width_range_start:width_range_start+tilesize, :]

def get_min_cut_patch(texture, tilesize, overlapsize, to_fill, to_fill_mask, patch=None):
    '''
    Returns a patch selected from the texture image that has been joined along the
    minimum cut with the existing content in to_fill

    Arguments:
    texture      - the texture image
    tilesize     - the desired tilesize
    overlapsize  - the overlap between tiles
    to_fill      - the patch that is being filled in the synthesized image
    to_fill_mask - a mask for to_fill marking which parts have been filled
    patch        - a patch that has been preselected

    Returns:
    A tilesize x tilesize section of texture
    '''

    # TODO: Implement this function
    # Choose a patch that minimizes the SSD, then join this patch with the existing content
    # using the minimum error boundary cut

    width_mask = to_fill_mask.shape[1]
    height_mask = to_fill_mask.shape[0]

    patch = get_ssd_patch(texture, tilesize, to_fill, to_fill_mask) 

    if np.sum(to_fill) == 0:
        return patch
    elif (to_fill_mask[0, width_mask-1] == 0 and to_fill_mask[height_mask-1, 0] == 1): #Just find cut to the left 


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
        #print(index)
        final_mask[y, 0:index, :] = 0

        while y >= 1:

            if index == 0:
                select_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1

                #next_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1]])
            elif index == M_matrix.shape[1]-1:
                select_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index-1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index-1

                #next_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index-1]])
            else:
                select_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1], M_matrix[y-1, index-1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1
                elif select_index == 2:
                    next_index = index - 1
                
                
                #next_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1], M_matrix[y-1, index-1]])

            final_mask[y-1, 0:next_index, :] = 0
            index = next_index
            y = y - 1

        return np.multiply(final_mask, patch) + np.multiply(1-final_mask, to_fill)
    

    elif (to_fill_mask[0, width_mask-1] == 1 and to_fill_mask[height_mask-1, 0] == 0): #Just find cut above

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

        #y = M_matrix.shape[0] - 1
        y = 0 

        column = M_matrix[:, y]
        index = np.argmin(column)
        #print(index)
        final_mask[0:index, y, :] = 0

        while y <= M_matrix.shape[1] - 2:

            if index == 0:
                select_index = np.argmin([M_matrix[index, y+1], M_matrix[index+1, y+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1

                #next_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1]])
            elif index == M_matrix.shape[0]-1:
                select_index = np.argmin([M_matrix[index, y+1], M_matrix[index-1, y+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index-1

                #next_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index-1]])
            else:
                select_index = np.argmin([M_matrix[index, y+1], M_matrix[index+1, y+1], M_matrix[index-1, y+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1
                elif select_index == 2:
                    next_index = index - 1
                
                
                #next_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1], M_matrix[y-1, index-1]])

            final_mask[0:next_index, y+1, :] = 0
            index = next_index
            y = y + 1

        return np.multiply(final_mask, patch) + np.multiply(1-final_mask, to_fill)
    
    elif (to_fill_mask[0, width_mask-1] == 1 and to_fill_mask[height_mask-1, 0] == 1):

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

        #y = M_matrix.shape[0] - 1
        y = 0 

        column = M_matrix[:, y]
        index = np.argmin(column)
        #print(index)
        final_mask[0:index, y, :] = 0

        while y <= M_matrix.shape[1] - 2:

            if index == 0:
                select_index = np.argmin([M_matrix[index, y+1], M_matrix[index+1, y+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1

                #next_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1]])
            elif index == M_matrix.shape[0]-1:
                select_index = np.argmin([M_matrix[index, y+1], M_matrix[index-1, y+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index-1

                #next_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index-1]])
            else:
                select_index = np.argmin([M_matrix[index, y+1], M_matrix[index+1, y+1], M_matrix[index-1, y+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1
                elif select_index == 2:
                    next_index = index - 1
                
                
                #next_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1], M_matrix[y-1, index-1]])

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
        #print(index)
        final_mask[y, 0:index, :] = 0

        while y >= 1:

            if index == 0:
                select_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1

                #next_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1]])
            elif index == M_matrix.shape[1]-1:
                select_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index-1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index-1

                #next_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index-1]])
            else:
                select_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1], M_matrix[y-1, index-1]])
                if select_index == 0:
                    next_index = index
                elif select_index == 1:
                    next_index = index + 1
                elif select_index == 2:
                    next_index = index - 1
                
                
                #next_index = np.argmin([M_matrix[y-1, index], M_matrix[y-1, index+1], M_matrix[y-1, index-1]])

            final_mask[y-1, 0:next_index, :] = 0
            index = next_index
            y = y - 1

        return np.multiply(final_mask, patch) + np.multiply(1-final_mask, to_fill)





# ------------------------TEXTURE FUNCTIONS------------------------

def image_texture(source, texture, outsize, tilesize, overlapsize, n_iter, mask, quiet):
    '''
    Outputs an image that looks like the source but is created from samples
    from the texture image
    '''

    adjsize = tilesize - overlapsize
    imout = np.zeros((math.ceil(outsize[0] / adjsize) * adjsize + overlapsize, math.ceil(outsize[1] / adjsize) * adjsize + overlapsize, source.shape[2]))

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

    width = texture.shape[1]
    height = texture.shape[0]

    width_mask = to_fill_mask.shape[1]
    height_mask = to_fill_mask.shape[0]

    max_width_integer = np.floor(width/tilesize)
    max_height_integer = np.floor(height/tilesize)

    width_range = range(int(max_width_integer))
    height_range = range(int(max_height_integer))

    source_gray = rgb2gray(source)

    patch_indices = []
    errors = []

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


            error1 = alpha * (reg_error + previous_error)
            error2 = ( 1 - alpha) * corr_error

            error = error1 + error2

            errors.append(error)
    
    errors, patch_indices = zip(*sorted(zip(errors, patch_indices)))
    N = 3

    errors_N = errors[0:N]
    patch_indices_N = patch_indices[0:N]

    select = np.random.randint(N)
    index_select = patch_indices_N[select]

    width_index = index_select[1]
    height_index = index_select[0]

    width_range_start = width_index*tilesize
    height_range_start = height_index*tilesize

    patch = texture[height_range_start:height_range_start+tilesize, width_range_start:width_range_start+tilesize, :]

    if np.sum(to_fill) == 0:
        return patch
    elif (to_fill_mask[0, width_mask-1] == 0 and to_fill_mask[height_mask-1, 0] == 1): #Just find cut to the left 


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
    

    elif (to_fill_mask[0, width_mask-1] == 1 and to_fill_mask[height_mask-1, 0] == 0): #Just find cut above

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



    