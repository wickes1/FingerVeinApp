from sklearn.metrics.pairwise import cosine_similarity
from .single_template import Single_Template_AverageAll

import numpy as np


def template_FirstFrame_CosineSimilarity(features, num_of_template):
    templateList = np.empty((num_of_template, 512))
    templateList[0] = features[0]
    features = features[1:]
    index = 1
    while index < num_of_template:
        diff_score = 1
        for feature in features:
            score = 0
            for i in range(index):
                x = np.expand_dims(templateList[i], axis=0)
                y = np.expand_dims(feature, axis=0)
                score = score + cosine_similarity(x, y)
            score /= len(templateList)
            if score < diff_score:
                diff_score = score
                templateList[index] = feature
        index = index + 1
    return templateList


def template_FirstFrame_CosineSimilarity_Average(features, num_of_template):
    templateList = template_FirstFrame_CosineSimilarity(features, num_of_template)
    return single_template_averageAll(templateList)


def k_means(features, num_of_template):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_of_template)
    kmeans.fit(features)
    centers = kmeans.cluster_centers_
    return centers


def k_medoids(features, num_of_template):
    from sklearn_extra.cluster import KMedoids

    kmedoids = KMedoids(n_clusters=num_of_template, metric="euclidean").fit(features)
    centers = kmedoids.cluster_centers_
    return centers


if __name__ == "__main__":
    features = np.random.rand(150, 512)
    import time

    for i in range(1, 6):
        start = time.time()
        while np.any(np.isnan(features)):
            features = np.random.rand(150, 512)
        templateList = template_FirstFrame_CosineSimilarity(features, i)
        print(templateList.shape, time.time() - start)
