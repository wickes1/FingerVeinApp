from sklearn.metrics.pairwise import cosine_similarity
from .single_template import Single_Template_AverageAll
import numpy as np


def FirstFrame_CosineSimilarity(features, num_of_template):
    templateList = np.empty((num_of_template, 512))
    templateList[0] = features[0]
    features = features[1:]
    index = 1
    count = 0
    fingerList = []
    fingerList.append(0)
    while index < num_of_template:
        finalCount = 0
        count = 0
        diff_score = 1
        for feature in features:
            count = count + 1
            score = 0
            for i in range(index):
                x = np.expand_dims(templateList[i], axis=0)
                y = np.expand_dims(feature, axis=0)
                score = score + cosine_similarity(x, y)
            score /= len(templateList)
            if score < diff_score and not count in fingerList:
                diff_score = score
                templateList[index] = feature
                finalCount = count
        index = index + 1
        fingerList.append(finalCount)
    return templateList


def FirstFrame_CosineSimilarity_Average(features, num_of_template):
    templateList = FirstFrame_CosineSimilarity(features, num_of_template)
    return Single_Template_AverageAll(templateList)


# All sklearn.cluster methods
#  Number of templates methods
def kMeansPreprocessed(features, num_of_template):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_of_template)
    kmeans.fit(features)
    centers = kmeans.cluster_centers_
    return centers


def kMeans(features, num_of_template):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_of_template)
    kmeans.fit(features)
    centers = kmeans.cluster_centers_
    return centers


def kMedoids(features, num_of_template):
    from sklearn_extra.cluster import KMedoids

    kmedoids = KMedoids(n_clusters=num_of_template, metric="euclidean").fit(features)
    centers = kmedoids.cluster_centers_
    return centers


def featureAgglomeration(features, num_of_template):
    from sklearn.cluster import FeatureAgglomeration

    agglo = FeatureAgglomeration(n_clusters=num_of_template).fit(features.T)
    features_reduced = agglo.transform(features.T)
    return features_reduced.T


def miniBatchKMeans(features, num_of_template):
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(n_clusters=num_of_template, batch_size=features.shape[0] // num_of_template).fit(features)
    centers = kmeans.cluster_centers_
    return centers


def spectralClustering(features, num_of_template):
    from sklearn.cluster import SpectralClustering

    clustering = SpectralClustering(n_clusters=num_of_template).fit(features)
    labels = clustering.labels_
    cluster_sum = []
    for label in np.unique(labels):
        cluster_sum.append(np.sum(features[np.where(labels == label)], axis=0))

    cluster_sum = np.array(cluster_sum)
    unique, counts = np.unique(labels, return_counts=True)
    center = []
    for i in range(len(unique)):
        center.append(np.divide(cluster_sum[i], counts[i]))

    return np.array(center)


# Other parameters methods
def dbscan(features):
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN().fit(features)
    labels = clustering.labels_
    cluster_sum = []
    for label in np.unique(labels):
        cluster_sum.append(np.sum(features[np.where(labels == label)], axis=0))

    cluster_sum = np.array(cluster_sum)
    unique, counts = np.unique(labels, return_counts=True)
    center = []
    for i in range(len(unique)):
        center.append(np.divide(cluster_sum[i], counts[i]))

    return np.array(center)


def optics(features):
    from sklearn.cluster import OPTICS

    clustering = OPTICS(min_samples=10).fit(features)
    labels = clustering.labels_
    cluster_sum = []
    for label in np.unique(labels):
        cluster_sum.append(np.sum(features[np.where(labels == label)], axis=0))

    cluster_sum = np.array(cluster_sum)
    unique, counts = np.unique(labels, return_counts=True)
    center = []
    for i in range(len(unique)):
        center.append(np.divide(cluster_sum[i], counts[i]))

    return np.array(center)


def meanShift(features):
    from sklearn.cluster import MeanShift

    clustering = MeanShift().fit(features)
    labels = clustering.labels_
    cluster_sum = []
    for label in np.unique(labels):
        cluster_sum.append(np.sum(features[np.where(labels == label)], axis=0))

    cluster_sum = np.array(cluster_sum)
    unique, counts = np.unique(labels, return_counts=True)
    center = []
    for i in range(len(unique)):
        center.append(np.divide(cluster_sum[i], counts[i]))

    return np.array(center)


if __name__ == "__main__":
    features = np.random.rand(150, 512)
    import time

    for i in range(1, 6):
        start = time.time()
        while np.any(np.isnan(features)):
            features = np.random.rand(150, 512)
        templateList = template_FirstFrame_CosineSimilarity(features, i)
        print(templateList.shape, time.time() - start)
