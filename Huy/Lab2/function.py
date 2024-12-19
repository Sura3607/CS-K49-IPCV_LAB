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

def padding(img,padd):
    h, w = img.shape
    padded_img = np.zeros((h + padd * 2, w + padd * 2), dtype=img.dtype)

    padded_img[padd:-padd, padd:-padd] = img
    padded_img[:padd, padd:-padd] = img[0, :]  # Dòng trên
    padded_img[-padd:, padd:-padd] = img[-1, :]  # Dòng cuối 
    padded_img[padd:-padd, :padd] = img[:, [0]]  # Cột trái
    padded_img[padd:-padd, -padd:] = img[:, [-1]]  # Cột phải 

    # Góc 
    padded_img[:padd, :padd] = img[0, 0]       # Góc trên-trái
    padded_img[:padd, -padd:] = img[0, -1]     # Góc trên-phải
    padded_img[-padd:, :padd] = img[-1, 0]     # Góc dưới-trái
    padded_img[-padd:, -padd:] = img[-1, -1]   # Góc dưới-phải
    return padded_img

def hor_div(padded_img, base_img): #gradient theo phương ngang
    imgHor_div = np.zeros(base_img.shape, dtype= np.int16)

    for i in range(base_img.shape[0]):
        for j in range(base_img.shape[1]):
            imgHor_div[i, j] = padded_img[i, j+1] - padded_img[i, j]

    return np.clip(imgHor_div, 0, 255).astype(np.uint8)

def ver_div(padded_img, base_img): #gradient theo phương dọc
    imgVer_div = np.zeros(base_img.shape, dtype= np.int16)

    for i in range(base_img.shape[0]):
        for j in range(base_img.shape[1]):
            imgVer_div[i,j] = padded_img[i +1, j] - padded_img[i,j]

    return np.clip(imgVer_div, 0, 255).astype(np.uint8)

def gradient_magnitude(hor_div, ver_div): #độ lớn gradient
    return np.clip(np.sqrt(hor_div**2 + ver_div**2) ,0 ,255).astype(np.uint8)

def binary_segmentation(img, threshold): #Phân đoạn nhị phân
    newImg = np.zeros(img.shape, dtype= img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > threshold:
                newImg[i,j] = 255
    return newImg

def bfs(img,visited,i,j):
    queue = deque([(i,j)])
    visited[i][j] = True
    pixel = []

    while queue:
        x, y = queue.popleft()
        pixel.append((x,y))
        for nx in range(x - 1, x + 2):  
            for ny in range(y - 1, y + 2): 
                if 0 <= nx < len(img) and 0 <= ny < len(img[0]):
                    if not visited[nx][ny] and img[nx,ny] == 0:
                        visited[nx][ny] = True
                        queue.append((nx, ny))
    return pixel                       


def all_components(img):
    rows, cols = img.shape
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    components = 0

    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and img[i, j] == 0:
                bfs(img, visited, i, j)
                components += 1
    return components