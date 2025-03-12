import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel, AutoTokenizer
from src.imgtocap.config.configuration import ConfigurationManager


class ImageEncoder(nn.Module):
    def __init__(self, resnet_weights):
        super().__init__()
        resnet = models.resnet50(weights=resnet_weights)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, images):
        with torch.no_grad():
            return self.resnet(images).squeeze(-1).squeeze(-1)


class TextDecoder(nn.Module):
    def __init__(self, tokenizer, text_model_name):
        super().__init__()
        self.gpt = AutoModel.from_pretrained(text_model_name)
        self.gpt.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            return self.gpt(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state


class MultimodalModel(nn.Module):
    def __init__(
        self, tokenizer_path, resnet_weights, text_model_name, out_features, dropout
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.image_encoder = ImageEncoder(resnet_weights)
        self.text_decoder = TextDecoder(self.tokenizer, text_model_name)
        self.projection = nn.Linear(2048, 768)

        self.fusion = nn.Sequential(
            nn.Linear(768 * 2, len(self.tokenizer)), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_decoder(input_ids, attention_mask)

        image_projection = (
            self.projection(image_features)
            .unsqueeze(1)
            .repeat(1, text_features.size(1), 1)
        )
        combined_features = torch.cat((image_projection, text_features), dim=-1)

        return self.fusion(combined_features)


class Model(MultimodalModel):
    def __init__(self, config):
        super().__init__(
            tokenizer_path=config.tokenizer_path,
            resnet_weights=config.model_params.resnet_weights,
            text_model_name=config.model_params.caption_model_name,
            out_features=config.model_params.out_features,
            dropout=config.model_params.dropout,
        )


if __name__ == "__main__":
    try:
        config = ConfigurationManager().get_model_config()
        model = Model(config)
        print("Model initialized successfully.")
    except Exception as e:
        raise RuntimeError("Model initialization failed.") from e
