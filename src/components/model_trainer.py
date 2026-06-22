import sys
from typing import Tuple
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data,load_object,save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ClassificationMetricArtifact, DataTransformationArtifact, ModelTrainerArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Initiated Training RandomForestClassifier with specialized parameters")

            #Splitting the data into train and test data features and target variables
            # x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1] 
            x_train = train[:, :-1]
            y_train = train[:, -1]

            x_test = test[:, :-1]
            y_test = test[:, -1]

            # Initialize RandomForestClassifier with specified parameters
            model = RandomForestClassifier(
                n_estimators=self.model_trainer_config._n_estimators,
                min_samples_split=self.model_trainer_config._min_samples_split,
                min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                max_depth=self.model_trainer_config._max_depth,
                criterion=self.model_trainer_config._criterion,
                random_state=self.model_trainer_config._random_state
            )

            #Fit the model
            logging.info("Parameters specified!, Model training going on")
            model.fit(x_train,y_train)
            logging.info("Model training done")

            # Predictions and Evaluations metrics
            # y_pred = model.predict(x_test)  
            # accuracy = accuracy_score(y_test, y_pred)
            # f1 = f1_score(y_test, y_pred)
            # recall = recall_score(y_test, y_pred)
            # precision = precision_score(y_test, y_pred



            # Predictions and Evaluation metrics
            y_prob = model.predict_proba(x_test)[:, 1]

            best_f1 = 0.0
            best_threshold = 0.5

            for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                temp_pred = (y_prob >= threshold).astype(int)

                current_precision = precision_score(y_test, temp_pred)
                current_recall = recall_score(y_test, temp_pred)
                current_f1 = f1_score(y_test, temp_pred)

                logging.info(
                    f"Threshold={threshold:.1f} | "
                    f"Precision={current_precision:.4f} | "
                    f"Recall={current_recall:.4f} | "
                    f"F1={current_f1:.4f}"
                )

                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_threshold = threshold

            logging.info(
                f"Best Threshold={best_threshold:.1f} | "
                f"Best F1={best_f1:.4f}"
            )

            # Final predictions using best threshold
            y_pred = (y_prob >= best_threshold).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)


            # Creating metric artifact
            # metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score= recall)
            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision=precision,
                recall=recall
            )
            return model, metric_artifact

        except Exception as e:
            raise MyException (e, sys) from e
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data

            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path) 
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Train-Test data loaded")

            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr,test=test_arr)
            logging.info("Model object and artifact loaded") 

            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded")

            #Check if model's accuracy meets the required/expected threshold
            if accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")
            
            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object = trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            
            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact = metric_artifact
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e