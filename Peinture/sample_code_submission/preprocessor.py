from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

class Preprocessor:

    def __init__(self):
        self.pca = PCA(n_components = 200)
        
            
    
    """
    
    Standardize the data with the module sklearn.preprocessing.StandardScaler
    
    Parameters
    ----------
    X (2D array): The data to standardize
    
    Returns
    -------
    2D array: The data standardized

    """
    def standardizing(self, X):
        scaler = StandardScaler()
        new_X = np.asfarray(X, dtype='float')
        return scaler.fit_transform(new_X)
    
    
    """Apply the 'fit' function of the module sklearn.decomposition.PCA on the data"""
    def fit(self, X):
        return self.pca.fit(X)
    
    
    """Apply the 'transform' function of the module sklearn.decomposition.PCA on the data"""
    def transform(self, X):
        return self.pca.transform(X)
    

    """
    
    Preprocess the data
    
    Parameters
    ----------
    X (2D array): The data to preprocess
    
    Returns
    -------
    2D array: The data, after it has been standardized and redimensionned into an array of shape (len(X), n_components)
    
    """
    def fit_transform(self, X):
        new_X = self.standardizing(X)
        return self.fit(new_X).transform(new_X)

