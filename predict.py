from typing import Dict

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from cog import BasePredictor, Path, Input
from PIL import Image

# те же классы, что в inference.py (если в твоём inference.py другие – поправь тут)
class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_image(image_path: str) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def get_probabilities(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    return probs * 100.0  # в проценты


class Predictor(BasePredictor):
    def setup(self):
        """Вызывается один раз при старте контейнера — загружаем модель."""
        # Файл model_85_nn_.pth должен лежать рядом с predict.py
        self.model = torch.load("model_85_nn_.pth", map_location="cpu")
        self.model.eval()
        self.model.to(device)

    def predict(
        self,
        image: Path = Input(description="Фото лица для определения формы"),
    ) -> Dict[str, float]:
        """Инференс по одному изображению."""

        img_tensor = preprocess_image(str(image)).to(device)

        with torch.inference_mode():
            logits = self.model(img_tensor)

        percentages = get_probabilities(logits)[0]

        # вернём словарь "класс -> процент"
        return {
            class_names[i]: float(percentages[i].item())
            for i in range(len(class_names))
        }
