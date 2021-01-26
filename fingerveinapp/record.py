from PIL import Image
import cv2
import numpy as np
from utils import load_device

# default value: user: 001, length: 15 length = 15s, finger: left_index
# length = 0 -> while True until press 'q'
def record(user, length, finger, path="./record/"):
    try:
        scanAPI, hDevice, pBuffer, img_size = load_device()
    except:
        print("Device not connected")
        return

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output = cv2.VideoWriter(path + "{}_{}.avi".format(user, finger), fourcc, 10.0, (200, 400))
    length = 999 if length == 0 else length * 10  # convert length to frame
    frameCount = 0
    print("Recording start")
    print("Place your finger in device and rotate steadily within a small angle")
    print("*" * 80)
    while length > 0:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if scanAPI.ftrScanGetImage2(hDevice, 2, pBuffer):
            img = Image.frombuffer("L", [img_size.nWidth, img_size.nHeight], pBuffer, "raw", "L", 0, 1)  # PIL gray image
            img = np.array(img)  # CV2 gray image
            brightness = calcBrightness(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            if brightness > 150:
                continue
            frameCount = frameCount + 1
            length = length - 1
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            output.write(img)
            position = (5, 15)
            cv2.rectangle(img, (0, 0), (140, 40), (0, 0, 0), thickness=cv2.FILLED)
            cv2.putText(img, user + "_" + finger, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, "Frame: %d" % frameCount, (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Recording", img)
        else:
            break
    cv2.destroyAllWindows()


def calcBrightness(img):
    # Return the Mean Brightness of a Frame
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[..., 2].mean()


if __name__ == "__main__":
    kwargs = {"user": "test", "frameRecord": 15, "finger": "left_index"}
    record(**kwargs)