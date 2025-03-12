import torch
from PIL import Image
from main import model, tokenizer
from torchvision import transforms
from src.imgtocap.logger import logger
from src.imgtocap.entity.entity import PredictionConfig


class Prediction:
    def __init__(self, config: PredictionConfig):
        """Initializes the Prediction class with configuration settings."""
        self.config: PredictionConfig = config
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model
        self.tokenizer = tokenizer

    def _preprocess_image(self, image_input: str | Image.Image) -> torch.Tensor:
        """Preprocesses an image by resizing, normalizing, and converting it to a tensor."""
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        try:
            image = (
                Image.open(image_input).convert("RGB")
                if isinstance(image_input, str)
                else image_input.convert("RGB")
            )
            return transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.exception(f"Error preprocessing image: {e}")
            raise

    def _generate_caption(self, image_tensor: torch.Tensor) -> str:
        """Generates a caption for the given image tensor using the trained model."""
        try:
            logger.info("Generating caption...")
            bos_token: int = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.bos_token
            )
            eos_token: int = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.eos_token
            )
            pad_token: int = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.pad_token
            )

            if bos_token is None or eos_token is None:
                raise ValueError("Tokenizer is missing required BOS or EOS tokens.")

            tokens = self.tokenizer(
                [self.tokenizer.bos_token],
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)
            input_ids: torch.Tensor = tokens["input_ids"]
            attention_mask: torch.Tensor = tokens["attention_mask"]

            generated_caption = []

            for _ in range(self.config.params.max_length):
                with torch.no_grad():
                    output = self.model(image_tensor, input_ids, attention_mask)
                    predicted_token: int = output[:, -1, :].argmax(dim=-1).item()

                    if predicted_token in {eos_token, pad_token}:
                        break

                    generated_caption.append(predicted_token)
                    input_ids = torch.cat(
                        (
                            input_ids,
                            torch.tensor([[predicted_token]], device=self.device),
                        ),
                        dim=-1,
                    )
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.ones((1, 1), dtype=torch.long, device=self.device),
                        ),
                        dim=-1,
                    )

            caption: str = self.tokenizer.decode(
                generated_caption, skip_special_tokens=True
            )
            logger.info(f"Caption generated: {caption}")
            return caption
        except Exception as e:
            logger.exception(f"Error generating caption: {e}")
            return ""

    def predict(self, image_input: str | Image.Image) -> str:
        """Processes an uploaded image and generates a caption."""
        try:
            logger.info("Processing uploaded image")
            image_tensor = self._preprocess_image(image_input)
            return self._generate_caption(image_tensor)
        except Exception as e:
            logger.exception(f"Prediction failed: {e}")
            return ""
