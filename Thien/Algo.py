import cv2
import numpy as np

def invert_color(image):
    return 255-image

def adjust_brightness(image, alpha):
    adjusted_image = image + alpha 
    adjusted_image = np.clip(adjusted_image, 0, 255) 
    return adjusted_image