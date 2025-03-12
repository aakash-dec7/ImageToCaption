import urllib.error
import zipfile
from pathlib import Path
import urllib.request as request
from src.imgtocap.logger import logger
from src.imgtocap.entity.entity import DataIngestionConfig
from src.imgtocap.config.configuration import ConfigurationManager


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Initialize DataIngestion with the provided configuration.
        """
        self.config: DataIngestionConfig = config
        self.root_dir: Path = Path(self.config.root_dir)

    def __download_file(self, url: str, destination: Path) -> None:
        """
        Download the file from the given URL if it does not already exist.
        """
        if destination.exists():
            logger.info(f"File already exists: {destination}")
            return

        try:
            logger.info(f"Downloading {url} to {destination}...")
            destination.parent.mkdir(parents=True, exist_ok=True)
            request.urlretrieve(url, str(destination))
            logger.info(f"Download successful: {destination}")
        except urllib.error.HTTPError as http_err:
            logger.error(
                f"HTTP error {http_err.code} while downloading {url}: {http_err}"
            )
            raise RuntimeError("HTTP error occurred during file download") from http_err
        except urllib.error.URLError as url_err:
            logger.error(f"URL error while accessing {url}: {url_err}")
            raise RuntimeError("URL error occurred during file download") from url_err
        except Exception as e:
            logger.exception(f"Unexpected error downloading {url}.")
            raise RuntimeError("Unexpected error occurred during file download") from e

    def __extract_zip(self, zip_path: Path) -> None:
        """
        Extract the contents of a zip file to the root directory.
        """
        if not zip_path.exists():
            logger.error(f"Zip file not found: {zip_path}")
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.root_dir)
            logger.info(f"Extracted: {zip_path}")
        except Exception as e:
            logger.exception(f"Error extracting {zip_path}: {e}")
            raise RuntimeError("Error occurred during zip extraction") from e

    def run(self) -> None:
        """
        Run the data ingestion pipeline: Download and extract image and caption datasets.
        """
        image_path: Path = self.root_dir / Path(self.config.image_url).name
        caption_path: Path = self.root_dir / Path(self.config.caption_url).name

        self.__download_file(self.config.image_url, image_path)
        self.__download_file(self.config.caption_url, caption_path)
        self.__extract_zip(image_path)
        self.__extract_zip(caption_path)


if __name__ == "__main__":
    try:
        data_ingestion_config: DataIngestionConfig = (
            ConfigurationManager().get_data_ingestion_config()
        )
        data_ingestion: DataIngestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.run()
    except Exception as e:
        logger.exception("Data ingestion pipeline failed.")
        raise RuntimeError("Data ingestion pipeline failed.") from e
