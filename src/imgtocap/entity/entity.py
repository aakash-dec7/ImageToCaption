from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    root_dir: Path
    image_url: str
    caption_url: str
    download_path: Path


@dataclass
class DataPreprocessingConfig:
    root_dir: Path
    image_path: Path
    train_ids_path: Path
    test_ids_path: Path
    caption_path: Path
    tokenizer_path: Path
    params: dict
    model_params: dict


@dataclass
class DataTransformationConfig:
    root_dir: Path
    train_datadict_path: Path
    test_datadict_path: Path
    params: dict


@dataclass
class ModelConfig:
    tokenizer_path: Path
    model_params: dict


@dataclass
class ModelTrainingConfig:
    root_dir: Path
    train_dataset_path: Path
    params: dict


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    test_dataset_path: Path
    metrics_path: Path
    params: dict
    repo_name: str
    repo_owner: str
    mlflow_uri: str


@dataclass
class PredictionConfig:
    model_path: Path
    tokenizer_path: Path
    params: dict
