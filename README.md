# Rendering 360º Images into Paintings
Final Project of Fall 2023 Brown course CSCI1290: Computational Photography.

## Project Goal
The goal of the project was to transform entire 360º images into artistic stylizations based on famous painting styles. We stylized the specific regions of interests, such as the sky, while preserving other parts of the original image. In addition to the stylizations, we added a dimension of interactivity, similar to Google Street View, where the image can be rotated 360 degrees.

This project is built on influential works in neural style transfer, drawing inspiration from "A Neural Algorithm of Artistic Style" by Gatys et al. \cite{Gatys}, Efros et al.'s "Image Quilting for Texture Synthesis and Transfer" \cite{Efros} and "Exploring the Structure of a Real-Time, Arbitrary Neural Artistic Stylization Network" by Ghiasi et al. \cite{Ghiasi}. Our project delves into both style and texture transfer methods. This involved implementing neural networks, including Convolutional Neural Networks (CNNs) and the TensorFlow model TF-Hub, and employing a patch-based texture transfer algorithm. 

To learn more details about the project, read csci1290_final_report.pdf, which is attached to the Github.

https://github.com/AllenDufort/csci1290_final/assets/77863478/6567f597-abba-4fa3-a701-488c9ed8cdc5

## To clone code:
`git clone https://github.com/AllenDufort/csci1290_final.git `

## To run code:
If you want to do style transfer using the neural networks, go to the Neural Network code section to learn how to run `style_transfer_final.ipynb`. 
If you want to do style transfer using patch-based algorithms, go to the Patch-based code to learn how to run `main.py`.

### Neural Network code
Run `style_transfer_final.ipynb` online at Google Colab or Jupyter Notebook. 
Make sure that the runtime type is python 3 and the hardware accelerator is changed to T4 GPU.
For input and style images, upload the images you want to use from your computer, or use one of our example images.

### Patch-based code
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
        -q, --quiet
                    stops images from displaying, optional argument

    Only the filename is needed for images, not the full path (i.e. 'toast.png' not '~/<dir1>/<dir2>/toast.png')

If no mask is given, entire image is rendered as painting. If you want to run main.py with a mask, and DON'T want to manually make your own mask, make a mask of the input image in `style_transfer_final.ipynb` using `def create_mask`. Make sure that the mask and the source images are the same exact size!

## Style Transfer Citation
All of the models and most of the code (excluding the masking) in `style_transfer_final.ipynb` was derived from this linked [TensorFlow style transfer tutorial](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/style_transfer.ipynb#scrollTo=g_nWetWWd_ns).
