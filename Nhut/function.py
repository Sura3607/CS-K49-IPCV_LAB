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

def megre_to_frame(R0,G0,B0, A0):
    frame = A0.copy()
    frame[:600, :600] = cv2.addWeighted(R0, 100/255.0, frame[:600, :600], 1 - 100/255.0, 0)
    frame[:600, 200:] = cv2.addWeighted(B0, 100/255.0, frame[:600, 200:], 1 - 100/255.0, 0)  
    frame[200:, :600] = cv2.addWeighted(G0, 100/255.0, frame[200:, :600], 1 - 100/255.0, 0)
    return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    