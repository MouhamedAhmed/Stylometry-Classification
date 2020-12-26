from feature_extractor import FeatureExtractor
from utils import read_data
from classical_classifiers import NBayesClassifier, LogisticRegressionClassifier
import numpy as np
import argparse


def main(classifier, feature):

    # read and split the data
    xtrain, ytrain, xvalid, yvalid = read_data()

    # extract TfIdf and Count Features based on N-Grams
    extractor = FeatureExtractor(xtrain, ytrain, xvalid, yvalid)

    # classical classifiers
    # Naive Bayes classifiers 
    if classifier == 'naive_bayes':
        if feature == 'TfIdf':
            # Tf-Idf
            NBayes_TfIdf_classifier = NBayesClassifier(xtrain = extractor.xtrain_TfIdf, ytrain = ytrain)
            print('vaidation accuracy of Naive Bayes classifier using TfIdf Feature = ', NBayes_TfIdf_classifier.classifier.get_accuracy(extractor.xvalid_TfIdf, yvalid))

        elif feature == 'NGrams':
            # NGrams
            NBayes_Count_classifier = NBayesClassifier(xtrain = extractor.xtrain_Count, ytrain = ytrain)
            print('vaidation accuracy of Naive Bayes classifier using Count Feature = ', NBayes_Count_classifier.classifier.get_accuracy(extractor.xvalid_Count, yvalid))

        elif feature == 'TfIdf_NGrams':
            # Tf-Idf and NGrams
            NBayes_TfIdf_Count_classifier = NBayesClassifier(xtrain = extractor.xtrain_TfIdf_Count, ytrain = ytrain)
            print('vaidation accuracy of Naive Bayes classifier using both TfIdf and Count Features = ', NBayes_TfIdf_Count_classifier.classifier.get_accuracy(extractor.xvalid_TfIdf_Count, yvalid))

        else:
            print('unsupported feature, choose one classifier from(TfIdf, NGrams, TfIdf_NGrams)')

    # Logistic Regression classifiers 
    elif classifier == 'logistic_regression':
        if feature == 'TfIdf':
            # Tf-Idf 
            LogReg_TfIdf_classifier = LogisticRegressionClassifier(xtrain = extractor.xtrain_TfIdf, ytrain = ytrain)
            print('vaidation accuracy of Logistic Regression classifier using TfIdf Feature = ', LogReg_TfIdf_classifier.classifier.get_accuracy(extractor.xvalid_TfIdf, yvalid))
        
        elif feature == 'NGrams':
            # NGrams
            LogReg_Count_classifier = LogisticRegressionClassifier(xtrain = extractor.xtrain_Count, ytrain = ytrain)
            print('vaidation accuracy of Logistic Regression classifier using Count Feature = ', LogReg_Count_classifier.classifier.get_accuracy(extractor.xvalid_Count, yvalid))
        
        elif feature == 'TfIdf_NGrams':
            #Tf-Idf and NGrams
            LogReg_TfIdf_Count_classifier = LogisticRegressionClassifier(xtrain = extractor.xtrain_TfIdf_Count, ytrain = ytrain)
            print('vaidation accuracy of Logistic Regression classifier using both TfIdf and Count Features = ', LogReg_TfIdf_Count_classifier.classifier.get_accuracy(extractor.xvalid_TfIdf_Count, yvalid))
        
        else:
            print('unsupported feature, choose one classifier from(TfIdf, NGrams, TfIdf_NGrams)')

    else:
        print('unsupported classifier, choose one classifier from(logistic_regression, naive_bayes)...')




if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-classifier', '--classifier', type=str, help='choose one classifier from(logistic_regression, naive_bayes), default is naive_bayes', default = 'naive_bayes')
    argparser.add_argument('-feature', '--feature', type=str, help='features that will be extracted before classification (TfIdf, NGrams, TfIdf_NGrams), default is NGrams', default = 'NGrams')
    args = argparser.parse_args()

    main(args.classifier, args.feature)       
