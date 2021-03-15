from sklearn import preprocessing
from enroll_method.single_template import Single_Template_FirstFrame, Single_Template_AverageAll
from enroll_method.multi_template import FirstFrame_CosineSimilarity, FirstFrame_CosineSimilarity_Average, kMeans, kMedoids, featureAgglomeration, miniBatchKMeans, spectralClustering
from enroll_method.multi_template import kMeansPreprocessed
from enroll_method.multi_template import dbscan, optics, meanShift

singleTemplateMethodList = []
mutliTemplateMethodList = [kMeans]
preprocessingMethodList = []
