from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
import numpy as np
class GridSearch:
    def __init__(self, classifier_pipeline, param_grid, xtrain, ytrain):
        self.classifier_pipeline = classifier_pipeline
        self.param_grid = param_grid

        self.model = GridSearchCV(estimator=classifier_pipeline, param_grid=param_grid,
                                  scoring='accuracy', verbose=1, n_jobs=-1, refit=True)
        
        self.model.fit(xtrain, ytrain)

        self.best_parameters = self.model.best_estimator_.get_params()

        # print best params
        self.print_best_params()

        # print accuracy on train
        print('training accuracy = ', self.get_accuracy(xtrain, ytrain))

    def print_best_params(self):
        print("Best parameters for", self.classifier_pipeline.steps[0][0], 'classifier are')
        for param_name in sorted(self.param_grid.keys()):
            print("\t%s: %r" % (param_name, self.best_parameters[param_name]))
    
    def predict(self,x):
        predictions = self.model.predict_proba(x)
        return predictions
    
    def get_accuracy(self, x, ytrue):
        yhat = self.predict(x)
        correct = 0
        for i in range(len(yhat)):
            if(np.argmax(yhat[i]) == ytrue[i]):
                correct += 1
        return correct/len(yhat)


# Naive Bayes classifier
class NBayesClassifier:
    def __init__(self, xtrain, ytrain):
        self.model_name = 'naive_bayes'
        self.model = MultinomialNB()
        self.pipeline = pipeline.Pipeline([(self.model_name, self.model)])
        self.alpha_grid = np.arange(0.001, 0.01, 0.001).tolist() + np.arange(0.01, 0.1, 0.01).tolist() + np.arange(0.1, 1.0, 0.1).tolist() + np.arange(1.0, 100, 10).tolist()
        self.param_grid = {self.model_name + '__alpha': self.alpha_grid}

        self.classifier = GridSearch(classifier_pipeline = self.pipeline, param_grid = self.param_grid, xtrain = xtrain, ytrain = ytrain)


# Logistic Regression classifier
class LogisticRegressionClassifier:
    def __init__(self, xtrain, ytrain):
        self.model_name = 'logistic_regression'
        self.model = LogisticRegression()
        self.pipeline = pipeline.Pipeline([(self.model_name, self.model)])
        self.C_grid = np.arange(0.1, 5, 0.1).tolist()
        self.param_grid = {self.model_name + '__C': self.C_grid}

        self.classifier = GridSearch(classifier_pipeline = self.pipeline, param_grid = self.param_grid, xtrain = xtrain, ytrain = ytrain)

