from enroll_method.single_template import Single_Template_FirstFrame, Single_Template_AverageAll
from enroll_method.multi_template import FirstFrame_CosineSimilarity, FirstFrame_CosineSimilarity_Average, kMeans, kMedoids, featureAgglomeration, miniBatchKMeans, spectralClustering
from enroll_method.multi_template import kMeansPreprocessed
from enroll_method.multi_template import dbscan, optics, meanShift
from enroll_method.preprocessing import Preprocessing

singleTemplateMethodList = [Single_Template_FirstFrame, Single_Template_AverageAll]
# mutliTemplateMethodList = [kMeans, featureAgglomeration, miniBatchKMeans, spectralClustering, dbscan, optics, meanShift]
mutliTemplateMethodList = [FirstFrame_CosineSimilarity, kMeans, kMedoids, featureAgglomeration, miniBatchKMeans, spectralClustering, dbscan, optics, meanShift]
