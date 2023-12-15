'''
Project: Texture - main.py
CSCI 1290 Computational Photography, Brown U.
Written by Emanuel Zgraggen.
Converted to Python by Trevor Houchens.


Usage-

For help in running the code, run:
    
    python main.py help

To run the texture transfer:

    python main.py [-i <INPUT>] [-s <STYLE_SOURCE>] [-m <MASK>] [-q]

    options:
        -h, --help            
                        show this help message and exit
        -i INPUT, --input INPUT
                        Name of the input image the file that you want to restyle.
        -s STYLE_SOURCE, --style_source STYLE_SOURCE
                        Name of the texture image used to restyle the input
        -m MASK, --mask MASK  
                        Name of mask of input image, optional argument
        -q, --quiet     stops images from displaying, optional argument

    Only the filename is needed for images, not the full path (i.e. 'toast.png' not '~/<dir1>/<dir2>/toast.png')

'''

import os
import numpy as np
import cv2
from student import image_texture
import argparse


SOURCE_PATH = './data'
OUTPUT_PATH = './outputs/'
TRANSFER_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'transfer')


def transfer(args):
    '''
    Runs the texture transfer part of the project
    '''

    if args['input'] is None or args['style_source'] is None:
        print("You must provide an input image and a transfer source image for texture transfer")

    else:
        # Make output directory
        if not os.path.exists(TRANSFER_OUTPUT_PATH):
            os.mkdir(TRANSFER_OUTPUT_PATH)

        # read images
        input = cv2.imread(os.path.join(SOURCE_PATH, "images", args['input']))
        style_source = cv2.imread(os.path.join(SOURCE_PATH, "textures", args['style_source']))
        
        # convert to float in [0,1] range
        input = input.astype(np.float32) / 255.
        style_source = style_source.astype(np.float32) / 255.

        mask = cv2.imread(os.path.join(SOURCE_PATH, "images", args['mask']))
        quiet = args['quiet']

        print(f"Reproducing {args['input']} with the texture of {args['style_source']}")

        # hyperparameters to define patch (tile) size, overlap region, and number of iterations for passes.
        tilesize=78
        overlapsize=13
        n_iterations=2
        outsize = (input.shape[0], input.shape[1])

        output = image_texture(input, style_source, outsize, tilesize, overlapsize, n_iterations, quiet, mask)
        
        # write out
        input_name = args['input'].split(".")[0]
        style_source_name = args['style_source'].split(".")[0]
        cv2.imwrite(os.path.join(TRANSFER_OUTPUT_PATH, f"{input_name}_{style_source_name}_transfer.jpg"), output*255)


if __name__ == '__main__':
    # func_map = {'transfer': transfer}

    parser = argparse.ArgumentParser(description="CSCI1290 - Final Project Texture")
    # parser.add_argument("method", help="Name of the method to run ('transfer' or 'synthesis')")
    parser.add_argument("-i", "--input", help="Name of the input image the file that you want to restyle.")
    parser.add_argument("-s", "--style_source", help="Name of the texture image used to restyle the input")
    parser.add_argument("-m", "--mask", help="Name of mask of input image, optional argument", default=None)
    parser.add_argument("-q", "--quiet", help="stops images from displaying, optional argument", action="store_true")
    args = vars(parser.parse_args())

    transfer(args)

    # if args["method"] in func_map:
    #     func_map[args["method"]](args)
    # else:
    #     print(f"{args['method']} is not a supported command. Try using 'synthesis' or 'transfer'")
