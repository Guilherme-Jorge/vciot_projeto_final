import cv2
import numpy as np
from config.config import PATH_TO_IMAGES
from services.mahotas_mask.mahotas_mask_service import MahotasMaskService

path_list = [PATH_TO_IMAGES]
mask_service = MahotasMaskService(path_list)

img = cv2.imread("test_car.png")
# img = cv2.imread("test.jpg")
img_after_mask = mask_service.process_image(img)
contours, hierarchy = cv2.findContours(img_after_mask, 1, cv2.CHAIN_APPROX_NONE)
if len(contours) > 0 :
      c = max(contours, key=cv2.contourArea)
      M = cv2.moments(c)
      if M["m00"] !=0 :
          cx = int(M['m10']/M['m00'])
          cy = int(M['m01']/M['m00'])
          print("CX : "+str(cx)+"  CY : "+str(cy))
          cv2.circle(img, (cx,cy), 5, (255,255,255), -1)
cv2.drawContours(img, c, -1, (0,255,0), 1)
cv2.imshow("", img)
# cv2.imshow("", img_after_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# mask_service.save_image(img, PATH_TO_IMAGES)
mask_service.save_image_mask(img_after_mask, PATH_TO_IMAGES)
