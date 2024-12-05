import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

from IPython.display import Image

def brightness(img,alpha):
    b_img = img.astype(np.int16) + alpha 
    b_img = np.clip(b_img, 0, 255)
    return b_img.astype(np.uint8)

def multip_brightness(img,beta):
    b_img = img.astype(np.int16) * beta
    b_img = np.clip(b_img, 0, 255)
    return b_img.astype(np.uint8)
