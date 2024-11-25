import cv2
import numpy as np
import websocket
from config.config import (
    WEBSOCKET_HOST,
    HTTP_CAM_HOST,
    PATH_TO_IMAGES,
    X_STOP,
    Y_STOP,
    X_STRAIGHT,
    X_RIGHT,
    X_LEFT,
    Y_LEFT,
    Y_RIGHT,
    Y_STRAIGHT,
)
from services.mahotas_mask.mahotas_mask_service import MahotasMaskService

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
        try:
            _img_h, img_w = img.shape[:2]
            center_left = img_w * 0.25
            center_right = img_w * 0.75
            get_size_flag = True
        except:
            pass

    img_mask = mask_service.process_image(img)

    # Creates contours on image
    contours, hierarchy = cv2.findContours(img_mask, 1, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:        # Finds the area with the largest contour area
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)

        if M["m00"] != 0:
            # Finds the center of the contour area
            cx = M["m10"] // M["m00"]
            cy = M["m01"] // M["m00"]
            # print(f"CX : {cx}  CY : {cy}")

            # Logic to move the car
            if move_flag:
                if cx >= center_right:
                    print("Turn right")
                    # Move left wheel faster
                    ws.send(f'{{"x": {X_RIGHT}, "y": {Y_RIGHT}}}')
                if cx < center_right and cx > center_left:
                    print("Continue straigh")
                    # Move both wheels faster
                    ws.send(f'{{"x": {X_STRAIGHT}, "y": {Y_STRAIGHT}}}')
                if cx <= center_left:
                    print("Turn left")
                    # Move right wheel faster
                    ws.send(f'{{"x": {X_LEFT}, "y": {Y_LEFT}}}')
                
                # cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)
    else:
        print("No line detected!")
        ws.send(f'{{"x": {X_STOP}, "y": {Y_STOP}}}')

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

    # Move the car
    if key_input == ord("m"):
        ws.send(f'{{"x": {X_STOP}, "y": {Y_STOP}}}')
        move_flag = not move_flag

ws.close()
cv2.destroyAllWindows()
