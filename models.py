import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, data=None, train_points=None, val_points=None, test_points=None):
        if data is not None:
            self.data = data
            self.train_data, self.validate_data, self.test_data = self.data.create_model_datasets(train_points, val_points, test_points)
            # print(self.train_data)
            self.train_data = self.train_data.sample(frac=1).reset_index(drop=True)

            # print(self.train_data)
        

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def test(self):
        pass


class Logistic_Regression(Model):
    def __init__(self, data=None, train_points=120, val_points=0, test_points=60):
        super().__init__(data, train_points, val_points, test_points)
        self.model = linear_model.LogisticRegression()

    def fit(self, X=None, y=None):
        # print(self.train_data)
        # print(self.train_data)
        if X is None and self.train_data is not None:
            X = self.train_data[["x", "y", "z", "roll", "pitch", "yaw"]]
        else:
            raise ValueError("No data provided for fitting.")
        if y is None and self.train_data is not None:
            y = self.train_data["success"]
        else:
            raise ValueError("No target provided for fitting.")
        

        
        self.model.fit(X,y)

    def validate(self):
        pass

    def test(self, data=None):
        # print(self.test_data)
        data = self.test_data if data is None else data
        return self.model.score(data[["x", "y", "z", "roll", "pitch", "yaw"]], self.test_data["success"])

    def predict(self, X):
        return self.model.predict(X)
        
class SVM(Model):
    def __init__(self, data=None, train_points=120, val_points=0, test_points=60):
        super().__init__(data, train_points, val_points, test_points)
        self.model = svm.SVC()

    def fit(self, X=None, y=None):
        if X is None and self.train_data is not None:
            X = self.train_data[["x", "y", "z", "roll", "pitch", "yaw"]]
        else:
            raise ValueError("No data provided for fitting.")
        if y is None and self.train_data is not None:
            y = self.train_data["success"]
        else:
            raise ValueError("No target provided for fitting.")
        
        self.model.fit(X,y)


    def validate(self):
        pass

    def test(self):
        return self.model.score(self.test_data[["x", "y", "z", "roll", "pitch", "yaw"]], self.test_data["success"])

    def predict(self, X):
        return self.model.predict(X)

class Random_Forest(Model):
    def __init__(self, data=None, train_points=120, val_points=0, test_points=60):
        super().__init__(data, train_points, val_points, test_points)
        self.model = RandomForestClassifier()

    def fit(self, X=None, y=None):
        if X is None and self.train_data is not None:
            X = self.train_data[["x", "y", "z", "roll", "pitch", "yaw"]]
        else:
            raise ValueError("No data provided for fitting.")
        if y is None and self.train_data is not None:
            y = self.train_data["success"]
        else:
            raise ValueError("No target provided for fitting.")
        
        self.model.fit(X,y)

    def validate(self):
        pass

    def test(self, data=None):
        data = self.test_data if data is None else data
        return self.model.score(data[["x", "y", "z", "roll", "pitch", "yaw"]], data["success"])
    
    def predict(self, X):
        return self.model.predict(X)
    


def compare_models(models, data=None):
    results = {}
    for model in models:
        model.fit()
        accuracy = model.test()
        results[model.__class__.__name__] = accuracy
    return results