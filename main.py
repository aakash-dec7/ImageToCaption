import torch
from src.imgtocap.logger import logger
from transformers import AutoTokenizer
from src.imgtocap.component.model import Model
from src.imgtocap.config.configuration import ConfigurationManager


# Load prediction configuration
prediction_config = ConfigurationManager().get_prediction_config()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model and Tokenizer ONCE
try:
    logger.info(f"Loading model from: {prediction_config.model_path}")
    model = Model(config=ConfigurationManager().get_model_config()).to(device)
    model.load_state_dict(torch.load(prediction_config.model_path, map_location=device))
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load model: {e}")
    model = None

try:
    logger.info(f"Loading tokenizer from: {prediction_config.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(prediction_config.tokenizer_path)
    logger.info("Tokenizer loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load tokenizer: {e}")
    tokenizer = None
