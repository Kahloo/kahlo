import pickle
import numpy as np
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

choixClassifieur = 1

class classifieur (BaseEstimator):

    def __init__(self):
        self.is_trained = False
        
        if (choixClassifieur == 0):
            self.classifieur = MLPClassifier()
        elif (choixClassifieur == 1):
            self.classifieur = RandomForestClassifier(n_estimators = 1000)
        elif (choixClassifieur == 2):
            self.classifieur = MLPRegressor()
        elif (choixClassifieur == 3):
            self.classifieur = GaussianNB()
        elif (choixClassifieur == 4):
            self.classifieur = DecisionTreeClassifier()
        else:
            self.classifieur = AdaBoostClassifier()
            
    def fit(self, X, y):
        self.classifieur = self.classifieur.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        return self.classifieur.predict(X)
    
    def save(self, path="./"):
        pickle.dump(self.classifieur, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self