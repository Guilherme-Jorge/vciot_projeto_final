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

        return str(
            f"{current_time.year}-{current_time.month}-{current_time.day}_{current_time.hour}.{current_time.minute}.{current_time.second}.{current_time.microsecond}"
        )

    def process_image(self, img, ksize=(7, 7), sigmaX=0):
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_bw, ksize, sigmaX)

        threshold = mahotas.thresholding.rc(img_blur)
        threshold -= threshold/2
        # threshold = 39

        img_mask = img_bw.copy()
        img_mask[img_mask > threshold] = 255
        img_mask[img_mask < 255] = 0
        img_mask = cv2.bitwise_not(img_mask)
        img_mask = cv2.dilate(img_mask, None, iterations=1)

        return img_mask

    def save_image(self, img, path):
        ret = cv2.imwrite(f"{path}/{self._get_datetime_format()}.jpg", img)
        if not ret:
            print("Image: Picture not saved")
        else:
            print("Image: Picture saved")

    def save_image_mask(self, img_mask, path):
        ret = cv2.imwrite(f"{path}/{self._get_datetime_format()}_mask.jpg", img_mask)
        if not ret:
            print("Mask: Picture not saved")
        else:
            print("Mask: Picture saved")
