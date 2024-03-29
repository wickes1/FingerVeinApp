import numpy as np


def Single_Template_FirstFrame(features):
    return np.expand_dims(features[0], axis=0)


def Single_Template_AverageAll(features):
    return np.expand_dims(np.average(features, axis=0), axis=0)


if __name__ == "__main__":
    list = single_template_averageAll(np.array([[1, 2, 3], [1, 1, 1], [1, 1, 1]]))
    print(list)
    print(list.shape)