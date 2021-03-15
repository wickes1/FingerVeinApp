import numpy as np


class Preprocessing:
    class OutlierRemoval:
        def __init__(self):
            self.method_list = [self.iso, self.ee, self.lof, self.svm]
            self.index = 0

        def __iter__(self):
            return self.method_list[self.index]

        def __next__(self):
            if self.index < len(self.method_list):
                self.index += 1
                return self.method_list[self.index]
            else:
                raise StopIteration

        def __getitem__(self):
            return self.method_list[self.index]

        # def __init__(self, max_num):
        #     self.max_num = max_num
        #     self.index = 0

        # def __iter__(self):
        #     return self

        # def __next__(self):
        #     self.index += 1
        #     if self.index < self.max_num:
        #         return self.index
        #     else:
        #         raise StopIteration

        def iso(features):
            from sklearn.ensemble import IsolationForest

            clf = IsolationForest(n_estimators=10, warm_start=True)
            prediction = clf.fit_predict(features)
            features = features[prediction != -1]
            return features

        # Minimum Covariance Determinant
        def ee(features):
            from sklearn.covariance import EllipticEnvelope

            clf = EllipticEnvelope()
            prediction = clf.fit_predict(features)
            features = features[prediction != -1]
            return features

        # Local Outlier Factor
        def lof(features):
            from sklearn.neighbors import LocalOutlierFactor

            clf = LocalOutlierFactor()
            prediction = clf.fit_predict(features)
            features = features[prediction != -1]
            return features

        # One-Class SVM
        def svm(features):
            from sklearn.svm import OneClassSVM

            clf = OneClassSVM()
            prediction = clf.fit_predict(features)
            features = features[np.where(prediction != 1)]
            return features