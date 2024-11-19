import cv2
import numpy as np
from config.config import PATH_TO_IMAGES
from services.mahotas_mask.mahotas_mask_service import MahotasMaskService

path_list = [PATH_TO_IMAGES]
mask_service = MahotasMaskService(path_list)

img = cv2.imread("test.jpg")
img_after_mask = mask_service.process_image(img)

ret1 = mask_service.save_image(img, PATH_TO_IMAGES)
ret2 = mask_service.save_image_mask(img_after_mask, PATH_TO_IMAGES)

print(ret1)
print(ret2)
