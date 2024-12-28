import cv2
import numpy as np

def invert_color(image):
    return 255-image

def adjust_brightness(image, alpha):
    adjusted_image = image.astype(np.int16) + alpha 
    adjusted_image = np.clip(adjusted_image, 0, 255) 
    return adjusted_image.astype(np.uint8) 
def adjust_brightness2(image, beta):
    adjusted_image = image.astype(np.float32) * beta
    adjusted_image = np.clip(adjusted_image, 0, 255) 
    return adjusted_image.astype(np.uint8)