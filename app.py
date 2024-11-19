import cv2
import numpy as np
import websocket
from config.config import WEBSOCKET_HOST, HTTP_CAM_HOST, PATH_TO_IMAGES
from services.mahotas_mask.mahotas_mask_service import MahotasMaskService

path_list = [PATH_TO_IMAGES]
mask_service = MahotasMaskService(path_list)

ws = websocket.WebSocket()
ws.connect(f"ws://{WEBSOCKET_HOST}/ws")

ws_flag = False

try:
    cam = cv2.VideoCapture(HTTP_CAM_HOST)
    ret, img = cam.read()
    if not ret:
        ws_flag = True
except Exception as error:
    pass

bw_flag = False

while True:
    if ws_flag:
        ret, img = cam.read()
    else:
        img_orig = ws.recv()
        img_np = np.frombuffer(img_orig, dtype=np.uint8)
        img = cv2.imdecode(img_np, flags=1)

    img_after_mask = mask_service.process_image(img)

    if not bw_flag:
        cv2.imshow("Color car cam (P: save image) (C: change color) (M: move car) (Q: quit)", img)
    else:
        cv2.imshow("B&W car cam (P: save image) (C: change color) (M: move car) (Q: quit)", img_after_mask)

    key_input = cv2.waitKey(1) & 0xFF

    if key_input == ord("q"):
        break

    if key_input == ord("p"):
        mask_service.save_image(img, PATH_TO_IMAGES)
        mask_service.save_image_mask(img_after_mask, PATH_TO_IMAGES)

    if key_input == ord("c"):
        bw_flag = not bw_flag

    if key_input == ord("m"):
        ws.send('{"x": 210, "y": 128}')

ws.close()
cv2.destroyAllWindows()
