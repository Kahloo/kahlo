from classifieur import classifieur
import numpy as np
import unittest

#Unit tests for the 'classifieur' class.
class TestClassifieur(unittest.TestCase):
    
    def __init__(self):
        self.clf = classifieur().classifieur
     
    #Test the prediction of our classifier.
    def testClassification(self):
        X = [[0, -1], [1, 0], [0, 1], [-1, 0], [0,0]] # On déclare un tableau contenant des données à classifier (données d'entrainement).
        Y = [0, 0, 1, 1,0] # On déclare un tableau où se trouve les labels de ces données
        self.clf.fit(X, Y) # On effectue la phase d'apprentissage
        
        """On effectue des tests avec des ASSERT"""
        assert self.clf.predict([[-2,2]]) == np.array([1]) 
        assert self.clf.predict([[2,-2]]) == np.array([0])
        assert self.clf.predict([[-5,-3]]) == np.array([0])
        assert self.clf.predict([[0,3]]) == np.array([1])
               
        
if __name__=='__main__':
    unittest.main(argv =[''], verbosity=1, exit=False)
