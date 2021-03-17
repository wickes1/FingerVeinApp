import numpy as np
import pandas as pd


class Preprocessing:
    class OutlierRemoval:
        def __init__(self):
            # self.method_list = [func for func in dir(OutlierRemoval) if callable(getattr(OutlierRemoval, func)) and not func.startswith("__")]
            # self.method_list = [self.none, self.iso, self.ee, self.lof, self.svm, self.rrcf]
            self.method_list = [self.none]
            self.index = -1

        def __iter__(self):
            return self

        def __next__(self):
            if self.index < len(self.method_list) - 1:
                self.index += 1
                return self.method_list[self.index]
            else:
                raise StopIteration

        def none(self, features):
            return features

        def iso(self, features):
            from sklearn.ensemble import IsolationForest

            clf = IsolationForest(n_estimators=10, warm_start=True)
            prediction = clf.fit_predict(features)
            features = features[prediction != -1]
            return features

        # Minimum Covariance Determinant
        def ee(self, features):
            from sklearn.covariance import EllipticEnvelope

            clf = EllipticEnvelope()
            prediction = clf.fit_predict(features)
            features = features[prediction != -1]
            return features

        # Local Outlier Factor
        def lof(self, features):
            from sklearn.neighbors import LocalOutlierFactor

            clf = LocalOutlierFactor()
            prediction = clf.fit_predict(features)
            features = features[prediction != -1]
            return features

        # One-Class SVM
        def svm(self, features):
            from sklearn.svm import OneClassSVM

            clf = OneClassSVM()
            prediction = clf.fit_predict(features)
            features = features[np.where(prediction != 1)]
            return features

        # Robust Random Cut Forest
        def rrcf(self, features):
            import rrcf

            tree = rrcf.RCTree()
            for i, f in enumerate(features):
                tree.insert_point(f, index=i)
            df = []
            for i in range(len(features)):
                df.append(tree.codisp(i))
            df = pd.DataFrame(df, columns=["score"])
            # print(df.describe())
            low = df.quantile(0.25).item()
            high = df.quantile(0.75).item()
            features = features[(df["score"] > low) & (df["score"] < high)]
            return features