import os
import pickle
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from src.imgtocap.logger import logger
from src.imgtocap.utils.utils import update_yaml_file
from src.imgtocap.entity.entity import DataPreprocessingConfig
from src.imgtocap.config.configuration import ConfigurationManager


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig) -> None:
        """Initialize DataPreprocessing with configuration settings."""
        self.config: DataPreprocessingConfig = config
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.config.model_params.caption_model_name
        )
        self.tokenizer.add_special_tokens({"bos_token": "[SOS]", "eos_token": "[EOS]"})
        self.transform: transforms.Compose = self._define_transforms()
        self.captions_dict: dict[str, list[str]] = self._load_captions()

    def _define_transforms(self) -> transforms.Compose:
        """Define and return image transformations."""
        logger.info("Defining image transformations.")
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )

    def _load_captions(self) -> dict[str, list[str]]:
        """Load captions from the specified file."""
        logger.info(f"Loading captions from: {self.config.caption_path}")
        captions = {}
        try:
            with open(self.config.caption_path, "r", encoding="utf-8") as file:
                for line in file:
                    image_name, caption = line.strip().split("\t")
                    image_name = image_name.split("#")[0]
                    captions.setdefault(image_name, []).append(caption)
            logger.info(f"Loaded {len(captions)} caption entries.")
        except Exception as e:
            logger.exception(f"Error loading captions: {e}")
            raise
        return captions

    def _process_images(self, image_ids_path: str) -> list[dict[str, any]]:
        """Process images and their captions, returning a dataset."""
        logger.info(f"Processing images from: {self.config.image_path}")
        dataset = []
        try:
            with open(image_ids_path, "r", encoding="utf-8") as file:
                image_names = file.read().splitlines()

            for image_name in image_names:
                img_path = os.path.join(self.config.image_path, image_name)
                if image_name not in self.captions_dict or not os.path.exists(img_path):
                    continue
                try:
                    image = Image.open(img_path).convert("RGB")
                    transformed_image = self.transform(image)
                    caption = f"{self.tokenizer.bos_token} {self.captions_dict[image_name][0]} {self.tokenizer.eos_token}"
                    tokens = self.tokenizer(
                        caption,
                        padding="max_length",
                        max_length=self.config.params.max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    dataset.append(
                        {
                            "image": transformed_image,
                            "input_ids": tokens["input_ids"][0],
                            "attention_mask": tokens["attention_mask"][0],
                        }
                    )
                except Exception as e:
                    logger.warning(f"Skipping {image_name}: {e}")

            logger.info(f"Processed {len(dataset)} images.")
        except Exception as e:
            logger.exception(f"Error processing images: {e}")
            raise
        return dataset

    def _update_vocab_size(self) -> None:
        """Update vocabulary size in the configuration file."""
        logger.info("Updating vocabulary size.")
        try:
            vocab_size = len(self.tokenizer)
            update_yaml_file(
                "hyperparameters", "vocab_size", vocab_size, "config/params.yaml"
            )
            logger.info("Vocabulary size updated in params.yaml.")
        except Exception as e:
            logger.exception(f"Error updating vocabulary size: {e}")
            raise

    def _save_preprocessed_data(
        self, data: list[dict[str, any]], file_path: str
    ) -> None:
        """Save preprocessed data to a file."""
        try:
            with open(file_path, "wb") as file:
                pickle.dump(data, file)
            logger.info(f"Saved data to {file_path}")
        except Exception as e:
            logger.exception(f"Error saving data to {file_path}: {e}")
            raise

    def _save_tokenizer(self) -> None:
        """Save tokenizer for later use."""
        logger.info("Saving tokenizer.")
        try:
            os.makedirs(self.config.tokenizer_path, exist_ok=True)
            tokenizer_path = os.path.join(self.config.tokenizer_path, "tokenizer")
            self.tokenizer.save_pretrained(tokenizer_path)
            logger.info(f"Tokenizer saved at: {tokenizer_path}.")
        except Exception as e:
            logger.exception(f"Error saving tokenizer: {e}")
            raise

    def run(self) -> None:
        """Execute the data preprocessing pipeline."""
        logger.info("Starting data preprocessing.")
        train_data = self._process_images(self.config.train_ids_path)
        test_data = self._process_images(self.config.test_ids_path)
        self._update_vocab_size()
        self._save_preprocessed_data(
            train_data, os.path.join(self.config.root_dir, "train_datadict.pkl")
        )
        self._save_preprocessed_data(
            test_data, os.path.join(self.config.root_dir, "test_datadict.pkl")
        )
        self._save_tokenizer()
        logger.info("Data preprocessing completed successfully.")


if __name__ == "__main__":
    try:
        data_preprocessing_config = (
            ConfigurationManager().get_data_preprocessing_config()
        )
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.run()
    except Exception as e:
        logger.exception("Data preprocessing pipeline failed.")
        raise RuntimeError("Data preprocessing pipeline failed.") from e
