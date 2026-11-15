import cv2
import os

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




