from model.fvr import FVR
import os
import pickle
import cv2
from PIL import Image
import numpy as np


def feature_extract(vidPath="./record/", pklPath="./pickle/record_feature.pkl"):
    feature_extractor = FVR()
    vidList = []
    for file in os.listdir(vidPath):
        if file.endswith(".avi"):
            vidList.append(file.split(".")[0])
    vidList.sort()

    if not os.path.exists(pklPath):
        db = {}
    else:
        with open(pklPath, "rb") as f:
            db = pickle.load(f)
            vidList = list(set(vidList) - set(db.keys()))
            if not vidList:
                print("Empty video list, no record video or all video's features are already extracted")
                return

    for user in vidList:
        cap = cv2.VideoCapture(vidPath + user + ".avi")
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        embList = np.empty((0, 512))
        frameCount = 0
        while cap.isOpened():
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            ret, frame = cap.read()
            if ret == True:
                cgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # cv2 gray image
                pgray = Image.fromarray(cgray)  # PIL gray image
                emb, edges = feature_extractor.get_embedding(pgray)
                embList = np.append(embList, np.expand_dims(emb, axis=0), axis=0)
                frameCount = frameCount + 1
                print(
                    "{}\t{}/{}".format(
                        user,
                        frameCount,
                        length,
                    )
                )
            else:
                break
        db.update({user: embList})
        cap.release()
    with open(pklPath, "wb") as f:
        pickle.dump(db, f)


if __name__ == "__main__":
    feature_extract(vidPath="./record_second/", pklPath="./pickle/record_feature_second.pkl")
