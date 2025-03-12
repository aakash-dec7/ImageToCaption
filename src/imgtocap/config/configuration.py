from src.imgtocap.constant import *
from src.imgtocap.utils.utils import read_yaml, create_directories
from src.imgtocap.entity.entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    DataTransformationConfig,
    ModelConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
    PredictionConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_path=CONFIG_FILE_PATH,
        params_path=PARAMS_FILE_PATH,
        schema_path=SCHEMA_FILE_PATH,
    ):
        """
        Initializes the ConfigManager by reading configuration, parameters, and schema files.
        Creates necessary directories for storing artifacts.
        """

        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.schema = read_yaml(schema_path)

        create_directories(self.config.artifacts_root)

    ### Data Ingestion
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        ingestion_config = self.config.data_ingestion

        create_directories(ingestion_config.root_dir)

        return DataIngestionConfig(
            root_dir=ingestion_config.root_dir,
            image_url=ingestion_config.source.image_url,
            caption_url=ingestion_config.source.caption_url,
            download_path=ingestion_config.download_path,
        )

    ### Data Preprocessing
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        preprocessing_config = self.config.data_preprocessing
        preprocessing_params = self.params.hyperparameters
        preprocessing_model_params = self.params.model_params

        create_directories(preprocessing_config.root_dir)

        return DataPreprocessingConfig(
            root_dir=preprocessing_config.root_dir,
            image_path=preprocessing_config.image_path,
            train_ids_path=preprocessing_config.train_ids_path,
            test_ids_path=preprocessing_config.test_ids_path,
            caption_path=preprocessing_config.caption_path,
            tokenizer_path=preprocessing_config.tokenizer_path,
            params=preprocessing_params,
            model_params=preprocessing_model_params,
        )

    ### Data Transformation
    def get_data_transformation_config(self) -> DataTransformationConfig:
        transformation_config = self.config.data_transformation
        transformation_params = self.params.train_test_split

        create_directories(transformation_config.root_dir)

        return DataTransformationConfig(
            root_dir=transformation_config.root_dir,
            train_datadict_path=transformation_config.train_datadict_path,
            test_datadict_path=transformation_config.test_datadict_path,
            params=transformation_params,
        )

    ### Model
    def get_model_config(self) -> ModelConfig:
        model_config = self.config.model
        model_params = self.params.model_params

        return ModelConfig(
            tokenizer_path=model_config.tokenizer_path,
            model_params=model_params,
        )

    ### Model Training
    def get_model_training_config(self) -> ModelTrainingConfig:
        training_config = self.config.model_training
        training_params = self.params.hyperparameters

        create_directories(training_config.root_dir)

        return ModelTrainingConfig(
            root_dir=training_config.root_dir,
            train_dataset_path=training_config.train_dataset_path,
            params=training_params,
        )

    ### Model Evaluation
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        evaluation_config = self.config.model_evaluation
        evaluation_params = self.params.hyperparameters
        exp_tracking_config = self.config.experiment_tracking

        create_directories(evaluation_config.root_dir)

        return ModelEvaluationConfig(
            root_dir=evaluation_config.root_dir,
            model_path=evaluation_config.model_path,
            test_dataset_path=evaluation_config.test_dataset_path,
            metrics_path=evaluation_config.metrics_path,
            params=evaluation_params,
            repo_name=exp_tracking_config.repo_name,
            repo_owner=exp_tracking_config.repo_owner,
            mlflow_uri=exp_tracking_config.mlflow.uri,
        )

    ### Prediction
    def get_prediction_config(self) -> PredictionConfig:
        prediction_config = self.config.prediction
        prediction_params = self.params.hyperparameters

        return PredictionConfig(
            model_path=prediction_config.model_path,
            tokenizer_path=prediction_config.tokenizer_path,
            params=prediction_params,
        )
