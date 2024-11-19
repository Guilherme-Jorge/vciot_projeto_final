import os
import cv2
import mahotas
import datetime


class MahotasMaskService:
    def __init__(self, path_list):
        self.path_list = path_list
        self._create_dirs(path_list)
        
        
    def _create_dirs(self, dirs):
        for dir in dirs:
            try:
                os.mkdir(dir)
                print("Directory already created.")
            except:
                print("Directory already exist.")
            
    def _get_datetime_format(self):
        current_time = datetime.datetime.now()

        year = current_time.year
        month = current_time.month
        day = current_time.day
        hour = current_time.hour
        minute = current_time.minute
        second = current_time.second
        microsecond = current_time.microsecond

        return str(f"{year}-{month}-{day}_{hour}.{minute}.{second}.{microsecond}")


    def process_image(self, img, ksize=(7, 7), sigmaX=0):
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_bw, ksize, sigmaX)
        threshold = mahotas.thresholding.rc(img_blur)

        img_mask = img.copy()
        img_mask[img_mask > threshold] = 255
        img_mask[img_mask < 255] = 0
        img_mask = cv2.bitwise_not(img_mask)

        return img_mask
    
    def save_image(self, img, path):
        ret = cv2.imwrite(f"{path}/{self._get_datetime_format()}.jpg", img)
        if not ret:
            return False
        return True
    
    def save_image_mask(self, img_mask, path):
        ret = cv2.imwrite(f"{path}/{self._get_datetime_format()}_mask.jpg", img_mask)
        if not ret:
            return False
        return True

'''
import mahotas
import numpy as np
import cv2
img = cv2.imread('ponte.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converte
suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
T = mahotas.thresholding.rc(suave)
temp2 = img.copy()
temp2[temp2 > T] = 255
temp2[temp2 < 255] = 0
temp2 = cv2.bitwise_not(temp2)
resultado = np.vstack([
np.hstack([img, suave]),
np.hstack([temp, temp2]) ])
cv2.imshow("Binarização com método Otsu e Riddler-
Calvard", resultado)
cv2.waitKey(0)

'''