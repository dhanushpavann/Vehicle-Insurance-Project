import sys
import pandas as pd
from pandas import DataFrame

from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.yes:int = 0
        self.no:int = 1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_respone = self._asdict()
        return dict(zip(mapping_respone.values()), mapping_respone.keys())

class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self,  dataframe: DataFrame) -> DataFrame:
        """
        Function accepts preprocessed inputs (with all custom transformations already applied),
        applies scaling using preprocessing_object, and performs prediction on transformed features.
        """
        try:
            logging.info("Starting prediction process")


            #Step-1: Applying scaling transformations using the pretrained preprocessing object
            transformed_feature = self.preprocessing_object.transform(dataframe )

            #Step-2: Perform prediction using the trained model
            logging.info("Using the trained model for predictions")
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions
        except Exception as e:
            raise MyException(e, sys) from e
        
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
    