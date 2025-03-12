import os
import re
import yaml
import json
import torch
import shutil
import torch.nn as nn
from pathlib import Path
from typing import Union
from box import ConfigBox
from datetime import datetime
from torch.utils.data import Dataset
from ensure import ensure_annotations
from src.imgtocap.logger import logger


class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict
        self.keys = list(data_dict.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        sample = self.data[key]
        return sample["image"], sample["input_ids"], sample["attention_mask"]


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents as a ConfigBox for dot notation access."""
    if not path_to_yaml.exists():
        logger.error(f"YAML file not found: {path_to_yaml.resolve()}")
        raise FileNotFoundError(f"YAML file not found: {path_to_yaml.resolve()}")

    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            data = yaml.safe_load(yaml_file) or {}
            return ConfigBox(data)
    except Exception as e:
        logger.exception(f"Error reading YAML file {path_to_yaml.resolve()}: {str(e)}")
        raise RuntimeError(
            f"Error reading YAML file {path_to_yaml.resolve()}: {str(e)}"
        )


@ensure_annotations
def update_yaml_file(
    section: str, key: str, value: Union[str, int, float, bool, dict, list], path: str
):
    """Updates a nested key in the YAML file at the given path."""
    try:
        try:
            with open(path, "r") as file:
                data = yaml.safe_load(file) or {}
        except FileNotFoundError:
            data = {}

        if section not in data:
            data[section] = {}

        data[section][key] = value

        with open(path, "w") as file:
            yaml.dump(data, file, default_flow_style=False)

        logger.debug(f"YAML file '{path}' updated successfully.")
    except Exception as e:
        logger.exception(f"Error updating YAML file: {e}")


@ensure_annotations
def create_directories(path):
    """Creates directories if they don't exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.exception(f"Error creating directories: {path}")
        raise e


@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    """Saves data as a JSON file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            logger.debug(f"JSON file successfully saved: {path}")
    except (TypeError, Exception) as e:
        logger.exception(f"Error saving JSON file: {path}")
        raise e


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads data from a JSON file and returns it as a ConfigBox."""
    if not path.exists():
        logger.error(f"JSON file not found: {path}")
        raise FileNotFoundError(f"JSON file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.debug(f"JSON file successfully loaded: {path}")
            return ConfigBox(data)
    except (json.JSONDecodeError, Exception) as e:
        logger.exception(f"Error loading JSON file: {path}")
        raise e


@ensure_annotations
def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    filename="checkpoint.pth",
):
    """Saves model checkpoint including epoch, model state, and optimizer state."""
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )


@ensure_annotations
def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    filename="checkpoint.pth",
    num_epochs=None,
):
    """Loads a checkpoint and resumes training if available, otherwise starts fresh."""
    backup_dir = "checkpoint"
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_path = os.path.join(backup_dir, f"checkpoint--{timestamp}.pth.backup")

    if os.path.exists(filename):
        try:
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]

            if start_epoch > num_epochs:
                logger.warning("Training already completed!")
                user_choice = (
                    input("Training completed. Start from scratch? (Y/N): ")
                    .strip()
                    .lower()
                )
                if user_choice != "y":
                    logger.info("Exiting training.")
                    exit()

                shutil.move(filename, backup_path)
                logger.warning(f"Checkpoint moved to backup: {backup_path}")
                return 1

            logger.info(f"Resuming training from epoch {start_epoch}")
            return start_epoch

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            shutil.move(filename, backup_path)
            logger.warning(f"Corrupt checkpoint moved to backup: {backup_path}")
            logger.info("Restarting training from scratch...")
            return 1

    else:
        logger.info("No checkpoint found, starting training from scratch...")
        return 1


@ensure_annotations
def get_package_info():
    """Retrieves package name and version from setup.py."""
    try:
        with open("setup.py", "r") as f:
            setup_content = f.read()
    except FileNotFoundError:
        return "unknown", "unknown"

    name_match = re.search(r"name=['\"]([^'\"]+)['\"]", setup_content)
    version_match = re.search(r"version=['\"]([^'\"]+)['\"]", setup_content)

    project_name = name_match.group(1) if name_match else "unknown"
    version = version_match.group(1) if version_match else "unknown"

    return project_name, version
