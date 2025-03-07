import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def adaptiveThresholding():
    # Load the image
    root = os.path.dirname(os.path.abspath(__file__))
    image = cv.imread(os.path.join(root, 'resource', 'FotoShaleShaker.png'))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(gray, cmap='gray')
    plt.title('gray')
    
    # Apply adaptive thresholding
    plt.subplot(122)
    block_size = 7
    c = 2
    threshold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, c)
    plt.imshow(threshold, cmap='gray')
    plt.title('adaptive thresholding')

    plt.show()

if __name__ == '__main__':
    adaptiveThresholding()