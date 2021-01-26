import warnings

from numpy.core.shape_base import block

warnings.filterwarnings("ignore")
import os
import pickle
import cv2
from PIL import Image
from model.fvr import FVR
from utils import load_device
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def identify(threshold=0.8):
    try:
        scanAPI, hDevice, pBuffer, img_size = load_device()
    except:
        print("Device not connected")
        return

    if not os.path.exists("./pickle/enrolled_user.pkl"):
        raise FileNotFoundError("enrolled_user.pkl not found, pls enroll first.")
    else:
        with open("./pickle/enrolled_user.pkl", "rb") as f:
            enrolled_user = pickle.load(f)
        if len(enrolled_user) == 0:
            raise ValueError("Empty enroll list, no enroll user")

    feature_extractor = FVR()

    print("Identify Start")

    plotDict = {}
    df = pd.DataFrame(columns=enrolled_user.keys())
    plt.figure()
    frameCount = 0
    winVerify = "Identifying"
    winMethod = "Method Score Threshold: {} (Press Q to quit)".format(threshold)
    cv2.namedWindow(winVerify)
    cv2.namedWindow(winMethod)
    cv2.moveWindow(winVerify, 1000, 500)
    cv2.moveWindow(winMethod, 1200, 500)
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if scanAPI.ftrScanGetImage2(hDevice, 2, pBuffer):

            pgray = Image.frombuffer("L", [img_size.nWidth, img_size.nHeight], pBuffer, "raw", "L", 0, 1)  # PIL gray image
            cgray = np.array(pgray)  # CV2 gray image
            emb, edges = feature_extractor.get_embedding(pgray)
            plotDict = {}
            for method, item in enrolled_user.items():
                max_score = -1
                id_user = "Unkown"
                for user, features in item.items():
                    for feature in features:
                        score = sum(emb * feature)
                        if score > max_score:
                            max_score = score
                            if max_score > threshold:
                                id_user = user

                plotDict.update({method: {"user": id_user, "score": max_score}})

            img = np.array(cgray)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            color = (0, 255, 0)
            for x in range(edges.shape[1]):
                img = cv2.circle(img, (int(edges[0][x]), img_size.nHeight - x - 1), 0, color, 2)
                img = cv2.circle(img, (int(edges[1][x]), img_size.nHeight - x - 1), 0, color, 2)
            cv2.imshow(winVerify, img)
            img2 = np.zeros((512, 256 * 3, 3), np.uint8)
            img2.fill(0)
            position = (5, 15)
            # df_temp = pd.DataFrame()
            frameCount = frameCount + 1
            cv2.putText(img2, "Frame Count: {}".format(frameCount), position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            for method, item in plotDict.items():
                color = (0, 0, 255) if item["user"] == "Unkown" else (0, 255, 0)
                disp_text = "Method: {} ID: {} Score: {:.4f}".format(method, item["user"], item["score"])
                position = (5, position[1] + 20)
                cv2.putText(img2, disp_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # df_temp[method] = [item["score"]]
            cv2.imshow(winMethod, img2)

            # realtime score plot, severely decrease performance frame rate!
            # df = df.append(df_temp,ignore_index=True)
            # plt.clf()
            # for method, score in df.iteritems():
            #     plt.plot(score, "-", label=method)
            #     plt.legend(loc="lower right")
            # plt.show(block=False)

            # cv2.rectangle(img, (0, 0), (160, 100), (0, 0, 0), thickness=cv2.FILLED)
            # cv2.putText(img, disp_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # cv2.putText(img, "Score: %.2f" % max_score, (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # cv2.putText(img, "Frame count: %d" % frameCount, (position[0], position[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # cv2.putText(img, "Image Quality: {}".format(prediction), (position[0], position[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorPrediction, 2)


def update_line(hl, new_data):
    hl.set_xdata(np.append(hl.get_xdata(), new_data))
    hl.set_ydata(np.append(hl.get_ydata(), new_data))
    plt.draw()


if __name__ == "__main__":
    identify()