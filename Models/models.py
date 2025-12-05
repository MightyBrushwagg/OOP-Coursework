"""
Machine learning models for grasp success prediction.

This module provides abstract base class and concrete implementations of
machine learning models for predicting grasp success based on gripper
position and orientation features.
"""

import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract base class for machine learning models.
    
    Defines the interface that all grasp prediction models must implement.
    Handles data splitting and provides abstract methods for training and evaluation.
    
    Attributes:
        data (Data): Data object containing training/validation/test sets.
        train_data (pd.DataFrame): Training dataset.
        validate_data (pd.DataFrame): Validation dataset.
        test_data (pd.DataFrame): Test dataset.
    """
    def __init__(self, data=None, train_points=None, val_points=None, test_points=None):
        """
        Initialize the model with data.
        
        Args:
            data (Data, optional): Data object containing simulation results.
            train_points (int, optional): Number of training points.
            val_points (int, optional): Number of validation points.
            test_points (int, optional): Number of test points.
        """
        if data is not None:
            self.data = data
            # Split data into train/validation/test sets
            self.train_data, self.validate_data, self.test_data = \
                self.data.create_model_datasets(train_points, val_points, test_points)
            # Shuffle training data for better learning
            self.train_data = self.train_data.sample(frac=1).reset_index(drop=True)

    @abstractmethod
    def fit(self):
        """
        Train the model on training data.
        
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def validate(self):
        """
        Evaluate the model on validation data.
        
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def test(self):
        """
        Evaluate the model on test data.
        
        Must be implemented by subclasses.
        """
        pass


class Logistic_Regression(Model):
    """
    Logistic Regression model for grasp success prediction.
    
    Uses scikit-learn's LogisticRegression for binary classification
    (success/failure) based on gripper position and orientation features.
    """
    def __init__(self, data=None, train_points=120, val_points=0, test_points=60):
        """
        Initialize Logistic Regression model.
        
        Args:
            data (Data, optional): Data object. Defaults to None.
            train_points (int, optional): Number of training points. Defaults to 120.
            val_points (int, optional): Number of validation points. Defaults to 0.
            test_points (int, optional): Number of test points. Defaults to 60.
        """
        super().__init__(data, train_points, val_points, test_points)
        self.model = linear_model.LogisticRegression()

    def fit(self, X=None, y=None):
        """
        Train the logistic regression model.
        
        Args:
            X (pd.DataFrame, optional): Feature matrix. Uses train_data if None.
            y (pd.Series, optional): Target vector. Uses train_data if None.
            
        Raises:
            ValueError: If no data or target is provided.
        """
        if X is None and self.train_data is not None:
            X = self.train_data[["x", "y", "z", "roll", "pitch", "yaw"]]
        else:
            raise ValueError("No data provided for fitting.")
        if y is None and self.train_data is not None:
            y = self.train_data["success"]
        else:
            raise ValueError("No target provided for fitting.")
        
        self.model.fit(X, y)

    def validate(self):
        """
        Evaluate model on validation set.
        
        Returns:
            NotImplemented: Method to be implemented in future versions.
        """
        pass

    def test(self, data=None):
        """
        Evaluate model on test set.
        
        Args:
            data (pd.DataFrame, optional): Test data. Uses self.test_data if None.
            
        Returns:
            float: Test accuracy score.
        """
        data = self.test_data if data is None else data
        return self.model.score(
            data[["x", "y", "z", "roll", "pitch", "yaw"]], 
            self.test_data["success"]
        )

    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Feature matrix with columns [x, y, z, roll, pitch, yaw].
            
        Returns:
            numpy.ndarray: Predicted success values (0 or 1).
        """
        return self.model.predict(X)
        
class SVM(Model):
    """
    Support Vector Machine model for grasp success prediction.
    
    Uses scikit-learn's SVC (Support Vector Classifier) for binary classification
    of grasp success based on gripper position and orientation features.
    """
    def __init__(self, data=None, train_points=120, val_points=0, test_points=60):
        """
        Initialize SVM model.
        
        Args:
            data (Data, optional): Data object. Defaults to None.
            train_points (int, optional): Number of training points. Defaults to 120.
            val_points (int, optional): Number of validation points. Defaults to 0.
            test_points (int, optional): Number of test points. Defaults to 60.
        """
        super().__init__(data, train_points, val_points, test_points)
        self.model = svm.SVC()

    def fit(self, X=None, y=None):
        """
        Train the SVM model.
        
        Args:
            X (pd.DataFrame, optional): Feature matrix. Uses train_data if None.
            y (pd.Series, optional): Target vector. Uses train_data if None.
            
        Raises:
            ValueError: If no data or target is provided.
        """
        if X is None and self.train_data is not None:
            X = self.train_data[["x", "y", "z", "roll", "pitch", "yaw"]]
        else:
            raise ValueError("No data provided for fitting.")
        if y is None and self.train_data is not None:
            y = self.train_data["success"]
        else:
            raise ValueError("No target provided for fitting.")
        
        self.model.fit(X, y)

    def validate(self):
        """
        Evaluate model on validation set.
        
        Returns:
            NotImplemented: Method to be implemented in future versions.
        """
        pass

    def test(self):
        """
        Evaluate model on test set.
        
        Returns:
            float: Test accuracy score.
        """
        return self.model.score(
            self.test_data[["x", "y", "z", "roll", "pitch", "yaw"]], 
            self.test_data["success"]
        )

    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Feature matrix with columns [x, y, z, roll, pitch, yaw].
            
        Returns:
            numpy.ndarray: Predicted success values (0 or 1).
        """
        return self.model.predict(X)

class Random_Forest(Model):
    """
    Random Forest model for grasp success prediction.
    
    Uses scikit-learn's RandomForestClassifier for binary classification
    of grasp success. Random forests are ensemble methods that combine
    multiple decision trees for robust predictions.
    """
    def __init__(self, data=None, train_points=120, val_points=0, test_points=60):
        """
        Initialize Random Forest model.
        
        Args:
            data (Data, optional): Data object. Defaults to None.
            train_points (int, optional): Number of training points. Defaults to 120.
            val_points (int, optional): Number of validation points. Defaults to 0.
            test_points (int, optional): Number of test points. Defaults to 60.
        """
        super().__init__(data, train_points, val_points, test_points)
        self.model = RandomForestClassifier()

    def fit(self, X=None, y=None):
        """
        Train the Random Forest model.
        
        Args:
            X (pd.DataFrame, optional): Feature matrix. Uses train_data if None.
            y (pd.Series, optional): Target vector. Uses train_data if None.
            
        Raises:
            ValueError: If no data or target is provided.
        """
        if X is None and self.train_data is not None:
            X = self.train_data[["x", "y", "z", "roll", "pitch", "yaw"]]
        else:
            raise ValueError("No data provided for fitting.")
        if y is None and self.train_data is not None:
            y = self.train_data["success"]
        else:
            raise ValueError("No target provided for fitting.")
        
        self.model.fit(X, y)

    def validate(self):
        """
        Evaluate model on validation set.
        
        Returns:
            NotImplemented: Method to be implemented in future versions.
        """
        pass

    def test(self, data=None):
        """
        Evaluate model on test set.
        
        Args:
            data (pd.DataFrame, optional): Test data. Uses self.test_data if None.
            
        Returns:
            float: Test accuracy score.
        """
        data = self.test_data if data is None else data
        return self.model.score(
            data[["x", "y", "z", "roll", "pitch", "yaw"]], 
            data["success"]
        )
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Feature matrix with columns [x, y, z, roll, pitch, yaw].
            
        Returns:
            numpy.ndarray: Predicted success values (0 or 1).
        """
        return self.model.predict(X)


def compare_models(models, data=None):
    """
    Train and compare multiple models on the same dataset.
    
    Trains each model, evaluates on test set, and returns accuracy scores
    for comparison.
    
    Args:
        models (list): List of Model instances to compare.
        data (Data, optional): Data object (currently unused). Defaults to None.
        
    Returns:
        dict: Dictionary mapping model class names to test accuracy scores.
    """
    results = {}
    for model in models:
        model.fit()  # Train the model
        accuracy = model.test()  # Evaluate on test set
        results[model.__class__.__name__] = accuracy
    return results