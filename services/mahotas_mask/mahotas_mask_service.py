import os
import cv2
import mahotas
import datetime


class MahotasMaskService:
    def __init__(self, path_list):
        self.current_img = None
        self.current_img_mask = None
        self.path_list = path_list
        self._create_dirs(path_list)
        
        
    def _create_dirs(self, dirs):
        for dir in dirs:
            try:
                os.mkdir(dir)
            except:
                pass
            
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

        self.current_img = img
        self.current_img_mask = img_mask

        return img_mask
    
    def save_image(self, path):
        ret = cv2.imwrite(f"{path}/{self._get_datetime_format()}.jpg", self.current_img)
        if not ret:
            return False
        return True
    
    def save_image_mask(self, path):
        ret = cv2.imwrite(f"{path}/{self._get_datetime_format()}_mask.jpg", self.current_img_mask)
        if not ret:
            return False
        return True
