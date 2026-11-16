import cv2
import os
import numpy as np

def clahe(img):
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)
    img_clahe=cv2.merge((y_clahe,cr,cb))
    img_fin=cv2.cvtColor(img_clahe, cv2.COLOR_YCrCb2BGR)
    return img_fin

def eqhist(img):
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)
    y = cv2.equalizeHist(y)
    img_eq=cv2.merge((y,cr,cb))
    equalized_img = cv2.cvtColor(img_eq, cv2.COLOR_YCrCb2BGR)
    return equalized_img

def adaptive_inpaint(img):
    blr = cv2.medianBlur(img, 15)

    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)

    at = cv2.adaptiveThreshold(np.array(255 - y), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 17)
    ia = np.array(255 - at)
    iv = cv2.adaptiveThreshold(ia, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 9)
    ib = cv2.subtract(iv, ia)

    bz = cv2.merge([ib, ib, ib])
    dsy = np.where(bz == (0, 0, 0), blr, img)
    threshold_value = 220
    mask = np.where(y > threshold_value, 255, 0).astype(np.uint8)
    result = cv2.inpaint(dsy, mask, 10, cv2.INPAINT_NS)
    return result
