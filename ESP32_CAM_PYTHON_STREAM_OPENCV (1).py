import cv2
import numpy as np
import requests
from ultralytics import YOLO   # YOLOv8

'''
INFO SECTION
- if you want to monitor raw parameters of ESP32CAM, open the browser and go to http://192.168.x.x/status
- command can be sent through an HTTP get composed in the following way http://192.168.x.x/control?var=VARIABLE_NAME&val=VALUE (check varname and value in status)
'''

# ESP32 URL
URL = "http://192.168.1.100"
AWB = True

# Face recognition and opencv setup
cap = cv2.VideoCapture(URL + ":81/stream")

# Load YOLOv8 pretrained model (có thể đổi sang yolov8m.pt hoặc yolov8l.pt nếu GPU đủ mạnh)
model = YOLO("yolov8n.pt")   # model nhẹ, phù hợp chạy CPU/ESP32 stream

def set_resolution(url: str, index: int=1, verbose: bool=False):
    try:
        if verbose:
            resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
            print("available resolutions\n{}".format(resolutions))

        if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            requests.get(url + "/control?var=framesize&val={}".format(index))
        else:
            print("Wrong index")
    except:
        print("SET_RESOLUTION: something went wrong")

def set_quality(url: str, value: int=1, verbose: bool=False):
    try:
        if value >= 10 and value <=63:
            requests.get(url + "/control?var=quality&val={}".format(value))
    except:
        print("SET_QUALITY: something went wrong")

def set_awb(url: str, awb: int=1):
    try:
        awb = not awb
        requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
    except:
        print("SET_AWB: something went wrong")
    return awb

if __name__ == '__main__':
    set_resolution(URL, index=8)

    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # YOLO detection
            results = model(frame, stream=True)  # dùng stream=True để nhanh hơn
            for r in results:
                for box in r.boxes:
                    # Tọa độ hộp
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Vẽ hộp
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{model.names[cls]} {conf:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

            cv2.imshow("YOLO Detection", frame)

            key = cv2.waitKey(1)

            if key == ord('r'):
                idx = int(input("Select resolution index: "))
                set_resolution(URL, index=idx, verbose=True)

            elif key == ord('q'):
                val = int(input("Set quality (10 - 63): "))
                set_quality(URL, value=val)

            elif key == ord('a'):
                AWB = set_awb(URL, AWB)

            elif key == 27:  # ESC
                break

    cv2.destroyAllWindows()
    cap.release()
