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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import joblib

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
    def __init__(self, data=None, train_points=None, val_points=None, test_points=None, shuffle=True):
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
                self.data.create_model_datasets(train_points, val_points, test_points, shuffle=shuffle)
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

    @abstractmethod
    def save_model(self, filename):
        """
        Save the trained model to a file.
        
        Args:
            filename (str): Path to the file where the model will be saved.
        """
        
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        """
        Load a trained model from a file.
        
        Args:
            filename (str): Path to the file from which the model will be loaded.
        """
        self.model = joblib.load(filename)


class Logistic_Regression(Model):
    """
    Logistic Regression model for grasp success prediction.
    
    Uses scikit-learn's LogisticRegression for binary classification
    (success/failure) based on gripper position and orientation features.
    """
    def __init__(self, data=None, train_points=120, val_points=0, test_points=60, shuffle=True):
        """
        Initialize Logistic Regression model.
        
        Args:
            data (Data, optional): Data object. Defaults to None.
            train_points (int, optional): Number of training points. Defaults to 120.
            val_points (int, optional): Number of validation points. Defaults to 0.
            test_points (int, optional): Number of test points. Defaults to 60.
        """
        super().__init__(data, train_points, val_points, test_points, shuffle=shuffle)
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
            print(self.train_data)
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
        
    def save_model(self, filename=None):
        filename = filename if filename else "logistic_regression_model.pkl"
        return super().save_model(filename)
    
class SVM(Model):
    """
    Support Vector Machine model for grasp success prediction.
    
    Uses scikit-learn's SVC (Support Vector Classifier) for binary classification
    of grasp success based on gripper position and orientation features.
    """
    def __init__(self, data=None, train_points=120, val_points=0, test_points=60, shuffle=True):
        """
        Initialize SVM model.
        
        Args:
            data (Data, optional): Data object. Defaults to None.
            train_points (int, optional): Number of training points. Defaults to 120.
            val_points (int, optional): Number of validation points. Defaults to 0.
            test_points (int, optional): Number of test points. Defaults to 60.
        """
        super().__init__(data, train_points, val_points, test_points, shuffle=shuffle)
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


    def save_model(self, filename=None):
        filename = filename if filename else "svm_model.pkl"
        return super().save_model(filename)

class Random_Forest(Model):
    """
    Random Forest model for grasp success prediction.
    
    Uses scikit-learn's RandomForestClassifier for binary classification
    of grasp success. Random forests are ensemble methods that combine
    multiple decision trees for robust predictions.
    """
    def __init__(self, data=None, train_points=120, val_points=0, test_points=60, n_estimators=100, shuffle=True):
        """
        Initialize Random Forest model.
        
        Args:
            data (Data, optional): Data object. Defaults to None.
            train_points (int, optional): Number of training points. Defaults to 120.
            val_points (int, optional): Number of validation points. Defaults to 0.
            test_points (int, optional): Number of test points. Defaults to 60.
        """
        super().__init__(data, train_points, val_points, test_points, shuffle=shuffle)
        self.model = RandomForestClassifier(n_estimators=n_estimators)

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
            print(X)
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

    def confusion(self):
        """
        Compute confusion matrix on test set.
        
        Returns:
            numpy.ndarray: Confusion matrix.
        """
        
        y_true = self.test_data["success"]
        y_pred = self.model.predict(self.test_data[["x", "y", "z", "roll", "pitch", "yaw"]])
        confuse = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=confuse,
                              display_labels=y_true.unique())

        return disp
        

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
    
    def predict(self, X, y):
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Feature matrix with columns [x, y, z, roll, pitch, yaw].
            
        Returns:
            numpy.ndarray: Predicted success values (0 or 1).
        """
        predictions = self.model.predict(X)
        accuracy = np.sum(predictions == y.values)
        return f"Predictions: {predictions}, \nAccuracy: {accuracy}/{len(y)}"

    def save_model(self, filename=None):
        filename = filename if filename else "random_forest_model.pkl"
        return super().save_model(filename)

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
        # model.save_model(f"Models/saved_models/{model.__class__.__name__}_model.pkl")
    return results