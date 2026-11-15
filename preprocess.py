import cv2
import os


basepath = 'src/Kvasir-SEG/images'
for entry in os.listdir(basepath):
    full_path = os.path.join(basepath, entry)
    if os.path.isfile(os.path.join(basepath, entry)):
        img = cv2.imread(full_path, 1)
        img_denoise = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        gray_img = cv2.cvtColor(img_denoise, cv2.COLOR_BGR2GRAY)
        dst = cv2.equalizeHist(gray_img)

        ycrcb_img = cv2.cvtColor(img_denoise, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

        cv2.imshow('chestie', img)
        cv2.waitKey()
        cv2.imshow('eliminare zgomot', img_denoise)
        cv2.waitKey()
        cv2.imshow('normalizare', dst)
        cv2.waitKey()
        cv2.imshow('normalizare2', equalized_img)
        cv2.waitKey()
        break
