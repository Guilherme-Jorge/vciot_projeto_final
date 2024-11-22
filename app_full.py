import os
import cv2
import datetime
import mahotas
import numpy as np
import websocket

WEBSOCKET_HOST = "192.168.4.1"
HTTP_CAM_HOST = f"https://{WEBSOCKET_HOST}:81/stream"
PATH_TO_IMAGES = "images/"
X_STOP = 210
X_LOW = 180
X_HIGH = 0
Y_STOP = 128
Y_LOW = 170
Y_HIGH = 255

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

path_list = [PATH_TO_IMAGES]
mask_service = MahotasMaskService(path_list)

ws = websocket.WebSocket()
ws.connect(f"ws://{WEBSOCKET_HOST}/ws")

# Values to stop the car
ws.send(f'{{"x": {X_STOP}, "y": {Y_STOP}}}')

ws_flag = True

# Tries to connect via HTTP to camera
try:
    cam = cv2.VideoCapture(HTTP_CAM_HOST)
    ret, img = cam.read()
    if ret:
        ws_flag = False
except Exception as error:
    pass

bw_flag = False
move_flag = False
get_size_flag = False

while True:
    # Connection to camera via WebSocket
    if ws_flag:
        img_orig = ws.recv()
        img_np = np.frombuffer(img_orig, dtype=np.uint8)
        img = cv2.imdecode(img_np, flags=1)

    # Connection to camera via HTTP
    else:
        ret, img = cam.read()

    # Finds image heigh x width first time it runs
    if not get_size_flag:
        img_h, _img_w = img.shape[:2]
        center_left = img_h * 0.25
        center_right = img_h * 0.75
        get_size_flag = True

    img_mask = mask_service.process_image(img)

    # Creates contours on image
    contours, hierarchy = cv2.findContours(img_mask, 1, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        # Finds the area with the largest contour area
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)

        if M["m00"] != 0:
            # Finds the center of the contour area
            cx = M["m10"] // M["m00"]
            cy = M["m01"] // M["m00"]
            print(f"CX : {cx}  CY : {cy}")

            # Logic to move the car
            if move_flag:
                if cx >= center_right:
                    print("Turn left")
                    # Move left wheel faster
                    ws.send(f'{{"x": {X_LOW}, "y": {Y_HIGH}}}')
                if cx < center_right and cx > center_left:
                    print("Straighten out")
                    # Move both wheels faster
                    ws.send(f'{{"x": {X_HIGH}, "y": {Y_HIGH}}}')
                if cx <= center_left:
                    print("Turn right")
                    # Move right wheel faster
                    ws.send(f'{{"x": {X_HIGH}, "y": {Y_LOW}}}')

                cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)

    cv2.drawContours(img, c, -1, (0, 255, 0), 1)

    if not bw_flag:
        cv2.imshow(
            "Color car cam (P: save image) (C: change color) (M: move car) (Q: quit)",
            img,
        )
    else:
        cv2.imshow(
            "B&W car cam (P: save image) (C: change color) (M: move car) (Q: quit)",
            img_mask,
        )

    key_input = cv2.waitKey(1) & 0xFF

    # Quit program
    if key_input == ord("q"):
        break

    # Save image
    if key_input == ord("p"):
        mask_service.save_image(img, PATH_TO_IMAGES)
        mask_service.save_image_mask(img_mask, PATH_TO_IMAGES)

    # Change output mode
    if key_input == ord("c"):
        bw_flag = not bw_flag

    # Move the car (TEMPORARY)
    if key_input == ord("m"):
        move_flag = not move_flag

ws.close()
cv2.destroyAllWindows()
