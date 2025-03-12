import os
import pickle
from src.imgtocap.logger import logger
from src.imgtocap.utils.utils import CustomDataset
from src.imgtocap.entity.entity import DataTransformationConfig
from src.imgtocap.config.configuration import ConfigurationManager


class DataTransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        """Initialize DataTransformation with configuration settings."""
        self.config = config

    def _load_pickle(self, file_path: str, data_type: str) -> dict:
        """Load a pickle file and return its data as a dictionary."""
        try:
            logger.info(f"Loading {data_type} data from {file_path}...")
            with open(file_path, "rb") as f:
                data_list = pickle.load(f)
            data_dict = {i: sample for i, sample in enumerate(data_list)}
            logger.info(
                f"{data_type.capitalize()} data loaded successfully with {len(data_dict)} samples."
            )
            return data_dict
        except FileNotFoundError:
            logger.exception(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.exception(f"Error loading {data_type} data: {e}")
            raise

    def _save_dataset(self, dataset: CustomDataset, file_name: str) -> None:
        """Save the dataset as a pickle file."""
        try:
            file_path = os.path.join(self.config.root_dir, file_name)
            os.makedirs(self.config.root_dir, exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(dataset, f)
            logger.info(f"Dataset saved successfully at {file_path}.")
        except Exception as e:
            logger.exception(f"Error saving dataset {file_name}: {e}")
            raise

    def run(self) -> None:
        """Execute the data transformation process."""
        try:
            logger.info("Starting data transformation process...")
            train_data = self._load_pickle(self.config.train_datadict_path, "train")
            test_data = self._load_pickle(self.config.test_datadict_path, "test")

            train_dataset, test_dataset = CustomDataset(train_data), CustomDataset(
                test_data
            )

            self._save_dataset(train_dataset, "train_dataset.pth")
            self._save_dataset(test_dataset, "test_dataset.pth")

            logger.info("Data transformation completed successfully.")
        except Exception as e:
            logger.exception("Data transformation process failed.")
            raise RuntimeError("Data transformation pipeline failed.") from e


if __name__ == "__main__":
    try:
        config = ConfigurationManager().get_data_transformation_config()
        data_transformation = DataTransformation(config=config)
        data_transformation.run()
    except Exception as e:
        logger.exception("Fatal error in data transformation pipeline.")
        raise RuntimeError("Data transformation pipeline terminated.") from e
