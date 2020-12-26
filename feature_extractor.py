import scipy.sparse as sp
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class FeatureExtractor:
    def __init__(self, xtrain, ytrain, xvalid, yvalid):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xvalid = xvalid
        self.yvalid = yvalid

        self.TfIdfVec = TfidfVectorizer(min_df=3,  max_features=None, 
                strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                stop_words = 'english')

        self.CountVec = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
                ngram_range=(1, 3), stop_words = 'english')

        # extract features
        self.xtrain_TfIdf, self.xvalid_TfIdf = self.fit_transform(self.TfIdfVec)
        self.xtrain_Count, self.xvalid_Count = self.fit_transform(self.CountVec)

        # combine both Tf-Idf and Count
        self.xtrain_TfIdf_Count = sp.hstack((self.xtrain_TfIdf, self.xtrain_Count))
        self.xvalid_TfIdf_Count = sp.hstack((self.xvalid_TfIdf, self.xvalid_Count))


    def fit_transform(self, extractor):
        # fit
        extractor.fit(self.xtrain)
        # transform
        xtrain_transformed = extractor.transform(self.xtrain)
        xvalid_transformed = extractor.transform(self.xvalid)

        return xtrain_transformed, xvalid_transformed



